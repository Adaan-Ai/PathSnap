# AI assistance record

- AI assistance was used to inspect the supplied project instructions and training/student-development data, implement EDA and data validation, candidate generation, route-sequence weak alignment, five candidate classifiers, Bayesian Random Forest tuning, Viterbi decoding, the prediction API/UI, experiment scripts, reports, and unit/smoke checks.
- The five classifier comparison, 10 Bayesian search trials, student-development evaluation, and decoder evaluation were executed against the supplied `train` and `student_test` partitions. Reported metrics come from those runs and are stored under `experiments/results/`; targets are explicitly described as route-derived weak labels.
- The trained Random Forest artifact is stored at `models/pathsnap_rf.joblib`. Source, documentation, reported results, and model artifact were committed and pushed to the `task/pathsnap-phase1-2` branch. Raw GPS datasets and coordinate-level EDA outputs were not committed.
- No evaluator-testing data or labels were accessed. The evaluator partition remains reserved for the invigilated session.
- Team members must review and understand the code, model settings, decoder assumptions, experiment design, and reported limitations. They should independently reproduce and be prepared to explain all submitted work; this log does not replace that responsibility.
