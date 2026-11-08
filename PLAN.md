# Current team plan (shared 6 November 2026)

This records the plan in the team message. Treat it as project context, not as an instruction to run jobs, spend credits, or change agreed quality limits without a separate decision.

1. A teammate creates the GitHub repo and uses Claude Ultra for initial iterations of the Gacrux decision matrix, then shares access.
2. Nandu gathers routing and coding datasets, establishes baselines, and runs later iterations on Google Colab compute units.
3. If model fine-tuning proceeds, share weights and rerun on Colab TPU units to investigate cost reduction.
4. Explore a difficulty estimator and routing at a finer task level, including assigning different models to frontend and backend work when a coding task can be split safely.
5. Keep the fast classifier and outcome-based retrieval as the main path. Investigate a per-enterprise local LLM router for uncertain cases, with shared company facts as a cached prefix and queued calls where latency permits.
6. Keep shared context to stable facts about the company, not a combined log of employees' requests. Retrieve similar past requests with model outcomes, within the same tenant, and use them to inform routing.
7. Retrain each company's decision matrix nightly from observed outcomes.
8. Investigate context growth and compression around 100k tokens, while protecting team privacy and output quality.

## Quality and cost decision to test

The team message proposes taking more routing risk to improve savings, including reconsidering tau and epsilon. In the Gacrux definition in `AGENTS.md`, **lower tau** routes more traffic to cheaper models and increases risk; higher tau does the opposite. Epsilon is the customer's allowed quality loss and should be agreed with that customer, not increased to make savings look better. Measure savings at a fixed epsilon first, then compare any separately agreed quality-cost tradeoffs.

The message also raises concern that existing routers' savings may be too small to matter. Establish this empirically against the best single model, kNN, Avengers-Pro, and the oracle, using cost per completed task and a quality-loss bound.
