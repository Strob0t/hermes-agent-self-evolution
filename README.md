# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

## Provider Configuration

The LLM provider is configured separately from the code — nothing is hardcoded to OpenAI. All model construction goes through `evolution/core/lm.py` (`make_lm`), which reads these environment variables:

| Variable | Purpose |
|----------|---------|
| `EVOLUTION_API_BASE` | Base URL of any OpenAI-compatible endpoint (Bifrost, LiteLLM proxy, vLLM, local gateway). |
| `EVOLUTION_API_KEY` | API key the endpoint expects. |
| `EVOLUTION_OPTIMIZER_MODEL` | Model id for GEPA reflections (e.g. `openai/gpt-4.1`). CLI `--optimizer-model` overrides it. |
| `EVOLUTION_EVAL_MODEL` | Model id for evaluation/judging. CLI `--eval-model` overrides it. |
| `COPILOT_GITHUB_TOKEN` | Optional: use GitHub Copilot models via the `copilot/<model>` prefix instead of a gateway. |

Example — route everything through a local Bifrost / LiteLLM gateway (keeps all traffic internal, no direct OpenAI calls):

```bash
cp .env.example .env            # then edit the values
export EVOLUTION_API_BASE=http://127.0.0.1:8080/v1
export EVOLUTION_API_KEY=changeme
export EVOLUTION_OPTIMIZER_MODEL=openai/gpt-4.1
export EVOLUTION_EVAL_MODEL=openai/gpt-4.1-mini

python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10 --eval-source synthetic
```

Explicit CLI flags (`--optimizer-model`, `--eval-model`) take precedence over the environment.

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | 🔲 Planned |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Planned |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Planned |

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars
3. **Caching compatibility** — No mid-conversation changes
4. **Semantic preservation** — Must not drift from original purpose
5. **PR review** — All changes go through human review, never direct commit

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
