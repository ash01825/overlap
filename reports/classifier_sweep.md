# Overlap Classifier Threshold Sweep Report

## 1. Corpus & Split Summary

- **Total Corpus Cases:** 240
- **Train Split (SHA-256 < 0.70):** 180 cases
- **Held-Out Split (SHA-256 >= 0.70):** 60 cases
- **Languages (10):** `bn, en, gu, hi, kn, ml, mr, pa, ta, te`
- **Grid Candidates Evaluated:** 72
- **Held-Out Target Evaluation:** **PASS**

## 2. Train vs Held-Out Performance

| Split / Configuration | Cases | Accuracy | FIR (Target <= 5.0%) | Missed Interrupt (Target <= 5.0%) | `t_verdict` p95 | Status |
|---|---:|---:|---:|---:|---:|---|
| **Train (Selected)** | 180 | 98.33% | 0.00% (0/60) | 0.00% (0/60) | 320.0 ms | PASS |
| **Held-Out (Baseline `Tuning()`)** | 60 | 96.67% | 0.00% (0/20) | 0.00% (0/20) | 321.0 ms | PASS |
| **Held-Out (Selected `Tuning`)** | 60 | 96.67% | 0.00% (0/20) | 0.00% (0/20) | 321.0 ms | **PASS** |

- **Generalization Gap (Train − Held-Out Accuracy):** +1.67%

### Findings

- Held-out FIR and missed-interruption rate both meet the <= 5.0% ARCHITECTURE §12 targets.

## 3. Recommended `Tuning` Default Changes

No changes to `Tuning` defaults recommended (`0` parameter deltas; baseline `Tuning()` achieves optimal train score and meets all held-out targets).

## 4. Held-Out Confusion Matrix

| Ground Truth \ Predicted | `BACKCHANNEL` | `INTERRUPT` | `SIDE_TALK` |
|---|---:|---:|---:|
| **`BACKCHANNEL`** | 20 | 0 | 0 |
| **`INTERRUPT`** | 0 | 20 | 0 |
| **`SIDE_TALK`** | 0 | 2 | 18 |

## 5. Held-Out Per-Language Breakdown

| Language | Cases | Accuracy | FIR | Missed Interrupt | BC / INT / ST |
|---|---:|---:|---:|---:|---:|
| `bn` | 2 | 100.0% | 0.0% | 0.0% | 1/0/1 |
| `en` | 8 | 100.0% | 0.0% | 0.0% | 4/3/1 |
| `gu` | 5 | 100.0% | 0.0% | 0.0% | 2/1/2 |
| `hi` | 9 | 88.9% | 0.0% | 0.0% | 3/2/4 |
| `kn` | 7 | 100.0% | 0.0% | 0.0% | 2/3/2 |
| `ml` | 3 | 100.0% | 0.0% | 0.0% | 0/2/1 |
| `mr` | 8 | 87.5% | 0.0% | 0.0% | 3/1/4 |
| `pa` | 4 | 100.0% | 0.0% | 0.0% | 1/1/2 |
| `ta` | 5 | 100.0% | 0.0% | 0.0% | 2/2/1 |
| `te` | 9 | 100.0% | 0.0% | 0.0% | 2/5/2 |

## 6. Grid-Search ROC / Threshold Table (Train Split — Top 15 & Bottom 5)

| Rank | `classify_min_ms` | `content_min_overlap_ms` | `bc_min_ms` | `assertive_db` | `sidetalk_db` | Train Acc | Train FIR | Train Missed | Deltas |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 250 | 250 | 200 | 3.0 | -9.0 | 98.33% | 0.00% | 0.00% | 0 |
| 2 | 220 | 250 | 200 | 3.0 | -9.0 | 98.33% | 0.00% | 0.00% | 1 |
| 3 | 250 | 220 | 200 | 3.0 | -9.0 | 98.33% | 0.00% | 0.00% | 1 |
| 4 | 250 | 250 | 180 | 3.0 | -9.0 | 98.33% | 0.00% | 0.00% | 1 |
| 5 | 250 | 250 | 200 | 2.0 | -9.0 | 98.33% | 0.00% | 0.00% | 1 |
| 6 | 250 | 250 | 200 | 3.0 | -7.5 | 98.33% | 0.00% | 0.00% | 1 |
| 7 | 220 | 220 | 200 | 3.0 | -9.0 | 98.33% | 0.00% | 0.00% | 2 |
| 8 | 220 | 250 | 180 | 3.0 | -9.0 | 98.33% | 0.00% | 0.00% | 2 |
| 9 | 220 | 250 | 200 | 3.0 | -7.5 | 98.33% | 0.00% | 0.00% | 2 |
| 10 | 250 | 220 | 180 | 3.0 | -9.0 | 98.33% | 0.00% | 0.00% | 2 |
| 11 | 250 | 220 | 200 | 2.0 | -9.0 | 98.33% | 0.00% | 0.00% | 2 |
| 12 | 250 | 220 | 200 | 3.0 | -7.5 | 98.33% | 0.00% | 0.00% | 2 |
| 13 | 250 | 250 | 180 | 2.0 | -9.0 | 98.33% | 0.00% | 0.00% | 2 |
| 14 | 250 | 250 | 180 | 3.0 | -7.5 | 98.33% | 0.00% | 0.00% | 2 |
| 15 | 250 | 250 | 200 | 2.0 | -7.5 | 98.33% | 0.00% | 0.00% | 2 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 68 | 220 | 250 | 200 | 4.0 | -10.5 | 97.22% | 0.00% | 1.67% | 3 |
| 69 | 220 | 220 | 200 | 4.0 | -10.5 | 97.22% | 0.00% | 1.67% | 4 |
| 70 | 250 | 220 | 180 | 4.0 | -10.5 | 97.22% | 0.00% | 1.67% | 4 |
| 71 | 220 | 250 | 180 | 4.0 | -10.5 | 97.22% | 0.00% | 1.67% | 4 |
| 72 | 220 | 220 | 180 | 4.0 | -10.5 | 97.22% | 0.00% | 1.67% | 5 |
