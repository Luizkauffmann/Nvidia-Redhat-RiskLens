"""LoRA SFT of Nemotron 3.5 Lightning on RiskLens trajectories (one GPU, BF16 base).

    python -m scripts.finetune_lora --check-data                 # tokenizer only, no GPU: lengths + loss mask
    python -m scripts.finetune_lora --smoke                      # ~10 steps: does LoRA on Lightning fit and train?
    python -m scripts.finetune_lora --data data/outputs/sft/nemotron_sft.jsonl --epochs 2

Input: SFT JSONL from src/evaluation/trajectories.py ({"messages", "tools", "metadata"}).
Loss is on assistant turns only (tool calls and answers); system, user and tool results are masked.
Rows are rendered with the model's chat template with thinking off ("<think></think>"), so serve
the adapter with enable_thinking=false.

LoRA targets attention (q/k/v/o_proj), Mamba in_proj (PEFT rejects out_proj) and the shared experts; the 128 routed
experts per layer stay frozen. Training deps: requirements-train.txt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
DATA = ROOT / "data" / "outputs" / "sft" / "mock_sft.jsonl"
# transformers 5 loads the checkpoint's "backbone." prefix as "model."; match either
TARGETS = r"(backbone|model)\.layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj|shared_experts\.up_proj|shared_experts\.down_proj)"
ASSISTANT_START, END = "<|im_start|>assistant\n", "<|im_end|>"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    for row in rows:
        for m in row["messages"]:
            for tc in m.get("tool_calls") or []:
                args = tc["function"].get("arguments")
                if isinstance(args, str):  # the chat template iterates arguments|items
                    tc["function"]["arguments"] = json.loads(args or "{}")
    return rows


def encode(tok, row: dict[str, Any], max_len: int) -> dict[str, list[int]] | None:
    """Render one conversation; labels = token ids inside assistant turns, -100 elsewhere."""
    text = tok.apply_chat_template(row["messages"], tools=row.get("tools") or None, tokenize=False,
                                   enable_thinking=False)
    spans, pos = [], 0
    while (start := text.find(ASSISTANT_START, pos)) != -1:
        end = text.find(END, start)
        end = len(text) if end == -1 else end + len(END)
        spans.append((start + len(ASSISTANT_START), end))
        pos = end
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [i if any(s <= a and b <= e for s, e in spans) else -100
              for i, (a, b) in zip(ids, enc["offset_mapping"])]
    return {"input_ids": ids, "labels": labels}


def check_data(tok, rows: list[dict[str, Any]], max_len: int) -> list[dict[str, list[int]]]:
    encoded, lengths, skipped = [], [], 0
    for row in rows:
        e = encode(tok, row, max_len)
        if e is None:
            skipped += 1
            continue
        encoded.append(e)
        lengths.append(len(e["input_ids"]))
    if not encoded:
        raise SystemExit(f"no rows fit max_len={max_len}")
    trained = [sum(l != -100 for l in e["labels"]) for e in encoded]
    lengths.sort()
    print(f"rows {len(encoded)} (skipped {skipped} > {max_len} tokens) | tokens min {lengths[0]} "
          f"median {lengths[len(lengths) // 2]} max {lengths[-1]} | trained tokens/row median "
          f"{sorted(trained)[len(trained) // 2]}")
    e = encoded[0]
    first = next(i for i, l in enumerate(e["labels"]) if l != -100)
    print("first supervised span:", repr(tok.decode([t for t in e["labels"][first:first + 60] if t != -100])))
    return encoded


def gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(ROOT / "data" / "outputs" / "lora" / "lightning-risklens"))
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="10 steps, report memory, save adapter")
    ap.add_argument("--check-data", action="store_true", help="tokenize only (no torch / GPU)")
    ap.add_argument("--no-kernels", action="store_true", help="skip Hub Mamba kernels (slow PyTorch path)")
    ap.add_argument("--cpu", action="store_true", help="allow running without CUDA (tiny test models only)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    encoded = check_data(tok, load_rows(Path(args.data)), args.max_len)
    if args.check_data:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    cuda = torch.cuda.is_available()
    if not cuda and not args.cpu:
        raise SystemExit("CUDA not available to torch (driver too old for this torch build?). Check "
                         "`nvidia-smi` and reinstall torch for that CUDA version; --cpu only for tiny test models.")
    t0 = time.time()
    # Hub kernels give the fast Mamba scan without building mamba-ssm; CPU uses the PyTorch path
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda" if cuda else "cpu",
                                                 use_kernels=cuda and not args.no_kernels)
    print(f"loaded in {time.time() - t0:.0f}s, GPU {gib(torch.cuda.memory_allocated()) if cuda else 'n/a'}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or 0
        return {"input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
                "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"]))
                                                for b in batch])}

    class TailLossTrainer(Trainer):
        """Logits only from the first supervised token on: rows are ~13k tokens (tool schemas) but only
        the tail is trained, and full-length logits (13k x 131k vocab) would cost >10 GB."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            first = int((labels != -100).int().argmax(dim=1).min())
            keep = labels.shape[1] - first + 1
            logits = model(**inputs, logits_to_keep=keep).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                     labels[:, first:].reshape(-1), ignore_index=-100)
            return loss

    out = Path(args.out + ("-smoke" if args.smoke else ""))
    steps = 10 if args.smoke else int(-(-len(encoded) // args.grad_accum) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        max_steps=10 if args.smoke else -1, num_train_epochs=args.epochs, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, steps // 20), bf16=True, logging_steps=1,
        save_strategy="no", report_to=[], remove_unused_columns=False, gradient_checkpointing=False)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer = TailLossTrainer(model=model, args=targs, train_dataset=encoded, data_collator=collate)
    # the loss is a per-row mean, so Trainer must divide it over gradient accumulation; otherwise it
    # logs (and backpropagates) the sum of the micro-steps. Set after __init__, which assigns it.
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train()
    peak = torch.cuda.max_memory_allocated() if cuda else 0
    total = torch.cuda.get_device_properties(0).total_memory if cuda else 0
    model.save_pretrained(out)
    tok.save_pretrained(out)
    summary = {"model": args.model, "data": args.data, "rows": len(encoded), "smoke": args.smoke,
               "steps": result.global_step, "train_loss": result.training_loss,
               "seconds": round(time.time() - t0), "peak_gpu": gib(peak), "gpu_total": gib(total),
               "rank": args.rank, "alpha": args.alpha, "max_len": args.max_len}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
