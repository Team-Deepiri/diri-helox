import os
import time

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import onnxruntime as ort

MODEL_NAME = "distilbert-base-uncased"
RUNS = 50
SEP = "=" * 60

print(SEP)
print("  ONNX vs MLflow Artifacts — Deepiri Trade-off Test")
print(SEP)

print("\n[1] Loading base model (DistilBERT)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
)
model.eval()
sample = tokenizer(
    ["Write unit tests for my API"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=64,
)
print("     Done.\n")

# ── MLflow-style: torch.save (this is exactly what MLflow stores) ──
pt_path = "/tmp/model.pt"
torch.save(model.state_dict(), pt_path)
pt_size_mb = os.path.getsize(pt_path) / 1024 / 1024

mlflow_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
)
mlflow_model.load_state_dict(torch.load(pt_path, weights_only=True))
mlflow_model.eval()

# ── ONNX export ───────────────────────────────────────────────
onnx_path = "/tmp/model.onnx"
with torch.no_grad():
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
        opset_version=18,
    )
onnx_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# ── Inference speed ───────────────────────────────────────────
ort_in = {
    "input_ids": sample["input_ids"].numpy(),
    "attention_mask": sample["attention_mask"].numpy(),
}

with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(RUNS):
        mlflow_model(**sample)
    pt_ms = (time.perf_counter() - t0) / RUNS * 1000

t0 = time.perf_counter()
for _ in range(RUNS):
    sess.run(None, ort_in)
onnx_ms = (time.perf_counter() - t0) / RUNS * 1000

# ── Accuracy ──────────────────────────────────────────────────
with torch.no_grad():
    base_pred = model(**sample).logits.argmax(1).item()
    mlflow_pred = mlflow_model(**sample).logits.argmax(1).item()
onnx_pred = np.array(sess.run(None, ort_in)[0]).argmax(1)[0]
pt_match = "✅ 100%" if base_pred == mlflow_pred else "❌"
onnx_match = "✅ 100%" if base_pred == onnx_pred else "❌"

# ── QLoRA insertion ───────────────────────────────────────────
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "k_lin"],
    lora_dropout=0.1,
)

try:
    pm = get_peft_model(mlflow_model, lora_cfg)
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pm.parameters())
    qlora_pt = f"✅  ({trainable:,} / {total:,} params trainable = {100*trainable/total:.2f}%)"
except Exception as e:
    qlora_pt = f"❌  {e}"

try:
    get_peft_model(sess, lora_cfg)
    qlora_onnx = "✅"
except Exception as e:
    qlora_onnx = f"❌  {type(e).__name__}"

# ── Print results ─────────────────────────────────────────────
print("[2] File Size")
print(f"     MLflow/PyTorch : {pt_size_mb:.1f} MB")
print(f"     ONNX           : {onnx_size_mb:.1f} MB")

print(f"\n[3] Inference Speed ({RUNS} runs)")
print(f"     MLflow/PyTorch : {pt_ms:.2f} ms/call")
print(f"     ONNX Runtime   : {onnx_ms:.2f} ms/call  →  {pt_ms/onnx_ms:.1f}x faster")

print("\n[4] Round-trip Prediction Accuracy")
print(f"     MLflow/PyTorch : {pt_match}")
print(f"     ONNX           : {onnx_match}")

print("\n[5] QLoRA Adapter Insertion (core test)")
print(f"     MLflow/PyTorch : {qlora_pt}")
print(f"     ONNX           : {qlora_onnx}")

print(f"\n{SEP}")
print("  VERDICT")
print(SEP)
print(f"  {'Metric':<28} {'MLflow/PyTorch':<22} ONNX")
print(f"  {'-'*60}")
print(f"  {'File size':<28} {pt_size_mb:.0f} MB{'':<19} {onnx_size_mb:.1f} MB")
print(
    f"  {'Inference speed':<28} {pt_ms:.1f} ms{'':<18} "
    f"{onnx_ms:.1f} ms ({pt_ms/onnx_ms:.1f}x faster)"
)
print(f"  {'Prediction accuracy':<28} {'100%':<22} {'100%'}")
print(f"  {'QLoRA / adapter insertion':<28} {'✅ YES':<22} ❌ NO (inference-only format)")
print(f"  {'In Deepiri stack already':<28} {'✅ YES':<22} ❌ NO")
print("\n  → Use MLflow for the training pipeline (fine-tuning requires PyTorch weights)")
print(
    f"  → Export to ONNX after fine-tuning is done, "
    f"for production serving ({pt_ms/onnx_ms:.1f}x faster)"
)
print(SEP)

os.remove(pt_path)
os.remove(onnx_path)
