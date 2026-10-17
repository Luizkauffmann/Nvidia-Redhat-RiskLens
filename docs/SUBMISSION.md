# Submission kit: team Residual Zero

Due **Sunday, October 18, 11:00 AM ET**, one submission per team, via the Airtable form linked on the event page.

## Checklist

| Item | Status |
|---|---|
| Team name | Residual Zero |
| Project description | draft below |
| Challenge selected | Multi-Agent Systems (bounties: Nemotron Lightning, Red Hat vLLM, NeMo Gym) |
| 3–5 min demo video, unlisted YouTube/Vimeo, core loop live | script below; record after the Brev run, add link to README |
| Repo link, public or judges invited | https://github.com/Luizkauffmann/Nvidia-Redhat-RiskLens (make sure it is public) |
| README: quick start | ✅ |
| README: tech stack & architecture diagram | ✅ (+ `docs/PROJECT_ARCHITECTURE.md`) |
| README: how to reproduce (env vars, API keys, sample .env) | ✅ (`.env.example`) |
| README: datasets / synthetic data + provenance | ✅ (`docs/DATA.md`, snapshot in `data/published/`) |
| README: known limitations & next steps | ✅ |
| Deployed URL or short screen capture | _record notebook 02 or `scripts.ask --trace` against vLLM_ |
| Team roster (names, roles, contacts) | _fill in the README table_ |
| Live Nemotron numbers (agent eval + Gym reward) | _run on Brev, add to README_ |
| Pre-event code import | _confirm with organizers in Discord that the labelled import is acceptable_ |

## Project description (draft, ~130 words)

> RiskLens AI explains why a bank's CECL credit-loss allowance changed. A team of agents, all running on one small open model (NVIDIA Nemotron 3.5 Lightning, 30B total / ~3B active, served with Red Hat vLLM on NVIDIA Brev), investigates the movement: an orchestrator delegates to Portfolio, Credit, Macro, Model Risk, Reconciliation, Quant and Reporting specialists, each limited to its own deterministic risk tools. The models never compute a number. Tools repair malformed arguments and return recovery hints, every figure in the final answer is checked against tool results, and an exact counterfactual attribution reconciles the explanation to the booked allowance to the cent. A NeMo Gym environment exposes the same tools and scores investigations against the hidden ground truth of a published synthetic 50,000-loan bank book.

## Demo video script (3–5 min)

Organisers' flow: team → pitch → live demo → how it was built → so what.

**1. Team (20–30 s).** "We are Residual Zero" + names and roles. "We built RiskLens, a team of small agents that explains credit-loss allowance movements."

**2. Pitch (30 s).** "Every month a bank's CECL allowance moves, and five teams spend days explaining why. We gave a team of Nemotron agents, a 3B-active model, specialist risk tools and one rule: never compute a number yourself."

**3. Live demo (90–120 s).** Notebook 02 (team section) or the terminal, model on vLLM/Brev, trace visible:
- "Why did our allowance increase in August?" → orchestrator delegates; $99.06M → $118.30M (+$19.24M); macro first, credit second; reconciliation PASS.
- "Which borrowers should I investigate?" → `CP_040408`: 3-notch downgrade, DSCR and LTV moves.
- "Could this just be because we changed models?" → no; CRE_PD_v4 in both months; weights moved 10% → 12% severe.
- Show one recovery (a repaired argument or an error hint followed by a correct retry) and the grounding badge.
- 20 s on NeMo Gym: the same tools as endpoints, `/verify` reward with its components.

**4. How we built it (60–90 s).** README architecture diagram. Deterministic engine + counterfactual attribution (exact, not plugged). Tool registry: validation, repair, hints, provenance. Team: one Nemotron base model, role prompts and allow-lists, findings carry their tool results. Grounding check, sandboxed Quant agent, Excel pack. Nemotron on vLLM: fits one GPU, fast enough for multi-turn delegation, keeps bank data on our own instance. Synthetic data with hidden ground truth and a Gym verifier, so we can score the agents.

**5. So what (20–30 s).** "Regulated teams can't trust a chatbot's numbers. RiskLens makes small, self-hosted models trustworthy by construction: every figure traceable to a tool call, every explanation reconciled to the booked allowance."
