import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


OUTPUT_DIR = Path("../dataset_processed")
PLOTS_DIR = Path("../dataset_processed")


def load_data() -> tuple[pd.DataFrame, dict[int, str]]:
    df = pd.read_csv(OUTPUT_DIR / "train.csv")
    with open(OUTPUT_DIR / "label_map.json") as f:
        label_map: dict[int, str] = {int(k): v for k, v in json.load(f).items()}
    df["label_name"] = df["label"].map(label_map)
    return df, label_map


def plot_samples(df: pd.DataFrame, label_map: dict[int, str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    rows = []
    for label_id, label_name in label_map.items():
        samples = df[df["label"] == label_id]["text"].head(2).tolist()
        for sample in samples:
            rows.append([label_name, sample[:120] + ("..." if len(sample) > 120 else "")])

    table = ax.table(
        cellText=rows,
        colLabels=["Class", "Sample text"],
        cellLoc="left",
        loc="center",
        colWidths=[0.15, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)
    plt.title("Sample Texts per Class", pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_1_samples.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_class_balance(df: pd.DataFrame, label_map: dict[int, str]) -> None:
    counts = df["label_name"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(counts.index, counts.values, color="steelblue")
    ax1.set_title("Class Distribution (bar)")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=30)

    ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Class Distribution (pie)")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_2_class_balance.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_length_dist(df: pd.DataFrame) -> None:
    df = df.copy()
    df["text_len"] = df["text"].astype(str).apply(len)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label_name, group in df.groupby("label_name"):
        ax.hist(group["text_len"], bins=50, alpha=0.6, label=label_name)

    ax.set_title("Text Length Distribution by Class")
    ax.set_xlabel("Character count")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_3_length_dist.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_wordclouds(df: pd.DataFrame, label_map: dict[int, str]) -> None:
    n = len(label_map)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = [axes] if n == 1 else axes.flatten()

    for ax, (label_id, label_name) in zip(axes, label_map.items()):
        texts = " ".join(df[df["label"] == label_id]["text"].astype(str).tolist())
        wc = WordCloud(width=600, height=300, background_color="white", max_words=80).generate(texts)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label_name)

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Word Clouds by Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_4_wordcloud.png", dpi=120, bbox_inches="tight")
    plt.close()


def print_stats(df: pd.DataFrame) -> None:
    df = df.copy()
    df["text_len"] = df["text"].astype(str).apply(len)
    total = len(df)
    counts = df["label_name"].value_counts()
    max_count = counts.max()
    min_count = counts.min()
    imbalance = max_count / min_count if min_count > 0 else float("inf")

    print(f"Total samples: {total}")
    print(f"Imbalance ratio: {imbalance:.2f}")
    print(f"Mean text length: {df['text_len'].mean():.0f} chars")
    print()
    for label_name, count in counts.items():
        mean_len = df[df["label_name"] == label_name]["text_len"].mean()
        print(f"  {label_name}: {count} samples, avg length {mean_len:.0f}")


def analyze() -> None:
    df, label_map = load_data()
    print_stats(df)
    plot_samples(df, label_map)
    plot_class_balance(df, label_map)
    plot_length_dist(df)
    plot_wordclouds(df, label_map)
    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    analyze()
