# Evaluation Scripts

## compare_models.py

`compare_models.py` compares trained model directories using quality and efficiency signals from:
- `training_info.json`
- `evaluation_report.json`

It prints:
- side-by-side summary table
- detailed per-model metrics
- ranked recommendation

It also writes a repeatable JSON report (default):
- `scripts/evaluation/model_comparison_report.json`

### Usage

```bash
python scripts/evaluation/compare_models.py \
  --models models/intent_classifier models/intent_classifier_candidate
```

### Emit JSON to stdout

```bash
python scripts/evaluation/compare_models.py \
  --models models/intent_classifier models/intent_classifier_candidate \
  --json
```

### Override report output path

```bash
python scripts/evaluation/compare_models.py \
  --models models/intent_classifier models/intent_classifier_candidate \
  --output-report scripts/evaluation/custom_model_comparison_report.json
```
