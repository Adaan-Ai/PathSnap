# Sequence decoder evaluation

## Method

The emission scorer is the selected Random Forest trained from the 80 training trips. For every student-test observation, candidate-level Random Forest probabilities are passed to the Viterbi decoder. Transition scores favor staying on an arc or moving to an arc whose start node matches the prior arc's end node, and penalize disagreement between estimated road distance and GPS displacement. A finite transition fallback handles noisy points and discontinuous route encodings. An observation with no candidate stays unmatched and splits the sequence.

Scoring targets are independently derived weak labels from the student-test `.route` sequences, aligned monotonically to GPS with a maximum 250 m projection distance. They are not manual per-observation truth. The `.route` files are accessed only in the evaluator scoring process; `PathSnapApplication.predict()` calls `load_trip(..., include_route_labels=False)` and does not require `.route` files.

## Results on student development set

| Metric | Classifier only | Viterbi decoded |
|---|---:|---:|
| Top-1 accuracy over 19,160 aligned observations | 0.8118 | **0.8948** |
| Mean per-trip transition continuity | 0.8991 | **0.9949** |

Candidate generation missed 91 of 19,160 aligned target arcs (0.47%), for both approaches. All 10 development trips were included. This evaluation uses a fixed 50 m radius and at most 8 candidates.

## Interpretation and limitations

Viterbi improved weak-label accuracy by 8.29 percentage points and substantially increased directed-road continuity on this development set. This is useful evidence for retaining the decoder, but it is not a final invigilator result. The alignment target can be ambiguous under GPS jitter, the transition model is a simple direct-adjacency model rather than shortest-path routing, and the result covers only the supplied 10 student-development trips. HMM penalties should be frozen before final evaluation, and the evaluator partition must remain unopened until the invigilated session.

Machine-readable measurements are in `experiments/results/decoder_student_test.json`. The code is in `src/pathsnap/decoder.py` and `experiments/evaluate_decoder.py`.
