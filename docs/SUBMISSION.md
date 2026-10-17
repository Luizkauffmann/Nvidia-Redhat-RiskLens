# Submission kit

Due **Sunday, October 18, 11:00 AM ET**, one submission per team, via the Airtable form linked on the event page.

## Checklist

| Item | Status |
|---|---|
| Team name | _fill in_ |
| Project description | draft below |
| Challenge selected | Tool-Calling (+ bounties: Nemotron Lightning, Red Hat vLLM) |
| 3–5 min demo video, unlisted YouTube/Vimeo, core loop live | script below; record after the Brev run |
| Repo link, public or judges invited | https://github.com/Luizkauffmann/Nvidia-Redhat-RiskLens (make sure it is public) |
| README: quick start | ✅ |
| README: tech stack & architecture diagram | ✅ (+ `docs/ARCHITECTURE.md`) |
| README: how to reproduce (env vars, API keys, sample .env) | ✅ (+ `.env.example`) |
| README: datasets / synthetic data + provenance | ✅ (+ `docs/DATA.md`, data in `data/published/`) |
| README: known limitations & next steps | ✅ |
| Deployed URL or short screen capture | _record a terminal capture of `scripts.ask --trace` against vLLM_ |
| Team roster (names, roles, contacts) | _fill in the README table_ |
| Live model evaluation numbers | _run `python -m scripts.evaluate --provider vllm`, paste into README_ |

## Project description (draft, ~120 words)

> RiskLens AI explains why a bank's CECL credit-loss allowance changed. A small open model, NVIDIA Nemotron 3.5 Lightning (30B total, ~3B active), served with vLLM on NVIDIA Brev, investigates the movement by calling 29 deterministic risk-analytics tools: allowance comparison, an exact counterfactual attribution (sequential, reverse and Shapley), PD/LGD/rating migration, borrower drill-down, macro-variable attribution, model-version checks and a reconciliation control. The model never computes a number. The tool layer repairs malformed arguments and returns errors with recovery hints, the agent accepts tool calls written as text, and every figure in the final answer is checked against tool results before it is shown. On a published synthetic 50,000-loan book with a hidden ground truth, the attribution reconciles to the booked change to the cent and recovers the injected causes.

## Demo video script (3–5 min)

Follow the organisers' flow: team → pitch → live demo → how it was built → so what.

**1. Team (20–30 s).** Names and roles. "We built RiskLens, a tool-calling investigator for credit-loss allowance."

**2. Pitch (30 s).** "Every month a bank's CECL allowance moves, and five teams spend days explaining why. We gave a 3B-active Nemotron model 29 risk tools and one rule: never compute a number yourself."

**3. Live demo (90–120 s).** Terminal, model on vLLM/Brev, `--trace` on:
- `python -m scripts.ask "Why did our allowance increase in August?" --provider vllm --trace` → tool calls appear, answer: $101.58M → $121.12M, macro first, reconciliation PASS.
- "Which borrowers should I investigate?" → CP_046990, 3-notch downgrade, DSCR and LTV moves.
- "Could this just be because we changed models?" → no; CRE_PD_v4 both months; weights moved 10% → 12% severe.
- "Does everything reconcile?" → residual $0.01, primary driver stable across three methods.
- Show one recovery: a bad date or portfolio name being repaired, or an error hint followed by a correct retry (visible in the trace / `data/outputs/tool_calls.jsonl`).

**4. How we built it (60–90 s).** Architecture diagram from the README. Deterministic engine + counterfactual attribution (exact, not plugged). Tool registry: validation, repair, hints, provenance. Agent loop: text-call parsing, duplicate suppression, compaction, grounding check with one revision, deterministic fallback. Nemotron 3.5 Lightning on vLLM: fits one GPU, fast enough for multi-turn tool loops, and keeps bank data on our own instance. Synthetic data with hidden ground truth so we can score the agent.

**5. So what (20–30 s).** "Regulated teams can't trust a chatbot's numbers. RiskLens makes a small, self-hosted model trustworthy by construction: every figure traceable to a tool call, every explanation reconciled to the booked allowance."
