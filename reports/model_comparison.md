# Candidate classifier comparison

## Experiment design

- Partition: the 80 supplied training trips only.
- Target: weak observation-level route arc IDs from monotonic route-to-track alignment (250 m maximum projection distance). This is not independently verified per-observation ground truth.
- Candidate settings: 50 m radius, at most 8 nearest directed arcs.
- Cross-validation: 5-fold `GroupKFold` by trip ID. Observations from one trip never appear in both the fit and validation folds.
- To keep the RBF SVM and KNN runs practical, each trip contributes at most 100 randomly selected aligned observations to fitting and a separate 50 to validation; seed 2026. The table reports means and standard deviations over folds. Candidate misses count as wrong in the coverage-adjusted Top-1 metric.
- Feature values are described in `src/pathsnap/candidates.py` and `experiments/train_compare.py`. Trip and arc IDs are not model features.

The full training partition had 99.83% weak-alignment coverage at the initial 250 m threshold. The 50 m/top-8 candidate generator missed 11.75% of aligned route arcs in the full observation-weighted diagnostic. The sampled, equal-trip CV subset had a 3.25% miss rate; these differ because the sample gives each trip a similar weight and the long trip `00000054` has unusually high noise.

## Five-model results

| Model | Algorithm family | Candidate macro F1 (mean ± fold SD) | Top-1, candidates covered | Top-1, all aligned samples | Log loss | Fit sec/fold | Inference sec/fold |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Linear probabilistic | 0.754 ± 0.014 | 0.739 | 0.715 | 0.367 | 0.08 | 0.002 |
| KNN (k=15) | Instance-based | 0.807 ± 0.021 | 0.731 | 0.708 | 0.443 | 0.16 | 1.878 |
| Random Forest | Bagged trees | 0.865 ± 0.014 | 0.804 | 0.778 | 0.197 | 1.60 | 0.060 |
| Histogram Gradient Boosting | Boosted trees | 0.838 ± 0.016 | 0.805 | 0.779 | 0.247 | 3.15 | 0.010 |
| RBF SVM | Kernel classifier | 0.812 ± 0.015 | 0.788 | 0.762 | N/A (uncalibrated decision score) | 41.38 | 7.076 |

Configurations: balanced Logistic Regression with `max_iter=300`; distance-weighted KNN with 15 neighbors; Random Forest with 120 trees, depth 14, leaf size 2 and balanced subsampling; HistGradientBoosting with 100 iterations, learning rate 0.08, 15 leaves and L2=1; scaled RBF SVM with `C=1` and balanced class weights.

## Interpretation and limits

Histogram Gradient Boosting and Random Forest were effectively tied on sampled coverage-adjusted Top-1 (0.779 and 0.778). Random Forest led on candidate macro F1 and probabilistic log loss; Histogram Gradient Boosting was faster at inference. These results do not establish a final model: the labels are route-derived weak targets, the per-trip validation sample is limited, one training trip has a very high candidate miss rate, and the separate student-development partition was not supplied. No evaluator data was used. Do not claim these scores as final route accuracy or use them to report an invigilator evaluation result.

The fold-level record is `experiments/results/model_comparison_folds.csv`; the aggregate record is `experiments/results/model_comparison_summary.json`; the chart is `experiments/results/model_comparison.png`. Re-run with `experiments/train_compare.py` to reproduce from the local training folder.

## Student-development evaluation

After the cross-validation comparison, all five models were fit on seeded samples of the 80 training trips (up to 100 aligned observations per trip) and evaluated on the separate 10-trip `student_test` partition. The reference `.route` files were read only by the scoring path; prediction inputs were loaded without route labels. Settings remained 50 m / top 8 so this is an initial held-out development comparison, not a tuned final result.

| Model | Top-1 accuracy, all aligned observations | Candidate macro F1 (mean per trip) | Log loss (mean per trip) | Fit seconds | Student inference seconds |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.749 | 0.755 | 0.366 | 0.13 | 0.04 |
| KNN (k=15) | 0.745 | 0.828 | 0.385 | 0.28 | 76.66 |
| Random Forest | **0.812** | **0.874** | **0.186** | 2.76 | 1.18 |
| Histogram Gradient Boosting | 0.810 | 0.848 | 0.230 | 4.99 | **0.29** |
| RBF SVM | 0.797 | 0.827 | N/A (uncalibrated) | 93.50 | 277.75 |

The candidate generator included the aligned route arc for 19,069 of 19,160 aligned observations (99.53%) in this student partition. Random Forest is the provisional lead; its Top-1 advantage over Histogram Gradient Boosting is only 0.23 percentage points, so do not treat this as a decisive final selection. HGB is much faster at inference. The test labels are still weak alignments from route sequences, not manually verified observation labels. The raw test route data and precise locations remain local and are not committed.

Machine-readable results are in `experiments/results/student_test_comparison.json` and `experiments/results/student_test_trip_results.csv`. Reproduce with `experiments/evaluate_student_test.py` using the local train and student-test paths.
