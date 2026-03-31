import json
import re
from pathlib import Path

import pandas as pd

from translit import translit as TRANSLIT_MAP


DATASET_DIR = Path(__file__).parent.parent / "dataset"
OUTPUT_DIR  = Path(__file__).parent.parent / "dataset_processed"

TRAIN_LABELS = {
    "probably_pro_israel":    "pro_great_israel",
    "probably_pro_palestine": "pro_so_called_palestine",
}

TEST_LABELS = {
    "certainly_pro_israel":    "pro_great_israel",
    "certainly_pro_palestine": "pro_so_called_palestine",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = "".join(TRANSLIT_MAP.get(ch, ch) for ch in text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_split(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    subset = df[df["label"].isin(label_map)].copy().reset_index(drop=True)
    subset["label"] = subset["label"].map(label_map)
    subset["text"] = subset["text"].astype(str).map(clean_text)
    return subset[subset["text"].str.len() > 0].reset_index(drop=True)


df_raw = pd.read_csv(DATASET_DIR / "results_full.csv")[["self_text", "label"]].rename(
    columns={"self_text": "text"}
).dropna(subset=["label"])

df_raw["label"] = df_raw["label"].str.strip()

train_df = build_split(df_raw, TRAIN_LABELS)
test_df  = build_split(df_raw, TEST_LABELS)

labels = ["pro_great_israel", "pro_so_called_palestine"]
label_to_int = {label: i for i, label in enumerate(labels)}
int_to_label = {i: label for label, i in label_to_int.items()}

train_df["label"] = train_df["label"].map(label_to_int)
test_df["label"]  = test_df["label"].map(label_to_int)

OUTPUT_DIR.mkdir(exist_ok=True)
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

with open(OUTPUT_DIR / "label_map.json", "w") as f:
    json.dump(int_to_label, f, ensure_ascii=False, indent=2)

print(f"train: {len(train_df)}, test: {len(test_df)}")
for label, idx in label_to_int.items():
    print(f"  {label}: train={( train_df['label'] == idx).sum()}, test={(test_df['label'] == idx).sum()}")
