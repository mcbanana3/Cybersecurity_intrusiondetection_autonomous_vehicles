# Experimental Results (auto-generated)

> All figures and numbers are computed from the real trained models on the held-out test split. This is a software simulation for research/education and is not production-grade security.

## Detection performance (test split)
- Binary accuracy: **0.990**
- Binary F1 (macro): **0.988**
- Binary ROC-AUC: **0.999**
- Binary PR-AUC (AP): **0.997**
- Multi-class accuracy: **0.979**
- Multi-class F1 (macro): **0.947**

## Top SHAP features (binary detector)
- `frame_count`: 0.0814
- `iat_mean`: 0.0781
- `frame_rate_hz`: 0.0694
- `top_id_fraction`: 0.0394
- `CHG_Status_grid_frequency_hz_min`: 0.0315
- `CHG_Status_grid_frequency_hz_max`: 0.0304
- `payload_entropy`: 0.0300
- `CHG_Status_grid_frequency_hz_range`: 0.0283

## Figures (results/figures/)
- confusion_matrix.png
- per_class_f1.png
- roc_curve.png
- pr_curve.png
- normal_vs_attack.png
- shap_importance.png

## Model comparison
See `results/model_comparison.csv`.