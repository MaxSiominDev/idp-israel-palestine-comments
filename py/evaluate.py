import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RUN = "deberta_base_1"

BASE          = Path(__file__).parent.parent
PROCESSED_DIR = BASE / "dataset_processed"

BATCH_SIZE = 64
MAX_LEN    = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_dir = BASE / "weights" / RUN

with open(PROCESSED_DIR / "label_map.json") as f:
    label_map: dict[int, str] = {int(k): v for k, v in json.load(f).items()}

NUM_LABELS = len(label_map)
tokenizer  = AutoTokenizer.from_pretrained(run_dir)

df     = pd.read_csv(PROCESSED_DIR / "test.csv")
texts  = df["text"].astype(str).tolist()
labels = df["label"].tolist()

model = AutoModelForSequenceClassification.from_pretrained(run_dir, local_files_only=True)
model.to(DEVICE)
model.eval()

per_class_correct = [0] * NUM_LABELS
per_class_total   = [0] * NUM_LABELS
correct = 0

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts  = texts[i : i + BATCH_SIZE]
    batch_labels = labels[i : i + BATCH_SIZE]

    enc = tokenizer(batch_texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(1).tolist()

    for p, l in zip(preds, batch_labels):
        per_class_total[l]   += 1
        per_class_correct[l] += int(p == l)
        correct += int(p == l)

    print(f"batch {i // BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}  samples {i + len(batch_texts)}/{len(texts)}")

acc = correct / len(texts) * 100
print(f"run: {RUN}  overall acc: {acc:.1f}%\n")

lines: list[str] = [f"# Test Evaluation\n", f"## {RUN}\n", f"Overall accuracy: **{acc:.1f}%**\n"]
lines.append("| Class | Correct | Total | Accuracy |")
lines.append("|-------|---------|-------|----------|")
for i, label_name in label_map.items():
    t = per_class_total[i]
    c = per_class_correct[i]
    lines.append(f"| {label_name} | {c} | {t} | {c/t*100:.1f}% |" if t else f"| {label_name} | 0 | 0 | — |")

(run_dir / "evaluate.md").write_text("\n".join(lines), encoding="utf-8")
print(f"saved -> weights/{RUN}/evaluate.md")
