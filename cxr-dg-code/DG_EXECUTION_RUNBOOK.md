# Final DG Execution Runbook (2-Source Protocol)

Operational handoff for the `1.0.0-dg-2source` study using NIH ChestX-ray14 and CheXpert sources.

## 1. Attach Inputs
Attach Kaggle inputs:
- Audited NIH/CheXpert source manifest + source images.
- Conservative Epic manifest (`EPIC_FINAL_MANIFEST.csv`) + Epic images.
- `cxr_pipeline.py`, `DG_Final_Execution.py`, `CXR_Master_Research_Pipeline.ipynb`.

Active source domains: NIH ChestX-ray14 and CheXpert. External target: Epic Chittagong.

## 2. Execution Sequence (8 Runs)
Per target protocol: `audit` -> `smoke` -> `lambda` -> `train` -> `evaluate` -> `post` -> `report`. The `lambda` phase locks the predeclared source-only value `0.1`; it is not a target-tuned sweep.

The Kaggle notebook uses bounded, explicitly reported training settings and one locked seed so the eight-run protocol fits within a normal Kaggle GPU session. A run is skipped only when its result file and referenced artifacts are complete.

### Run Matrix
| Runs | Target | Methods | Seed |
|---|---|---|---|
| 1--4 | Epic Chittagong | A, B, C, D | 42 |
| 5--6 | NIH ChestX-ray14 (LODO) | A, D | 42 |
| 7--8 | CheXpert (LODO) | A, D | 42 |

For the NIH and CheXpert LODO checks, only one source domain remains. Method D still tests MixStyle, but its CORAL term is mathematically inactive because there is no pair of source domains to align.

### CLI Commands Example
```bash
# 1. Audit
python DG_Final_Execution.py --phase audit --target "Epic Chittagong" --input-root /kaggle/input --output-root /kaggle/working/dg_suite

# 2. Smoke
python DG_Final_Execution.py --phase smoke --target "Epic Chittagong" --method A --input-root /kaggle/input --output-root /kaggle/working/dg_suite

# 3. Lambda (Lock the predeclared source-only CORAL value)
python DG_Final_Execution.py --phase lambda --target "Epic Chittagong" --input-root /kaggle/input --output-root /kaggle/working/dg_suite

# 4. Train (Repeat A, B, C, D for Epic; A, D for LODO)
python DG_Final_Execution.py --phase train --target "Epic Chittagong" --method A --input-root /kaggle/input --output-root /kaggle/working/dg_suite

# 5. Evaluate (Requires locked comparison models)
python DG_Final_Execution.py --phase evaluate --target "Epic Chittagong" --method A --input-root /kaggle/input --output-root /kaggle/working/dg_suite

# 6. Post (Grad-CAM and dynamic-range TFLite)
python DG_Final_Execution.py --phase post --target "Epic Chittagong" --method D --evaluation-path /kaggle/working/dg_suite/logs/DG-D-EPIC-S42/target_evaluation_epic_chittagong.json --output-root /kaggle/working/dg_suite

# 7. Report (Aggregate all 8 runs)
python DG_Final_Execution.py --phase report --output-root /kaggle/working/dg_suite --required-run-set minimum
