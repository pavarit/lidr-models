# news_sentiment

A placeholder package. The news-sentiment model (a second base learner that
ingests company-relevant headlines, scores them, and emits a per-ticker
sentiment feature/recommendation) is **Task 2** in the multi-model restructure.

- **Plan:** [`../../docs/plans/task-2-news-sentiment-model.md`](../../docs/plans/task-2-news-sentiment-model.md)
- **Architecture:** [`../../docs/adr/0001-multi-model-repo-architecture.md`](../../docs/adr/0001-multi-model-repo-architecture.md)
- **Data sources:** [`../../docs/research/data-sources.md`](../../docs/research/data-sources.md)

Until Task 2 lands, the package exposes only an empty `__init__.py` so that
the workspace builds; no signals, models, or configs are registered yet.
