# =============================================================================
# train_model.py — Benchmark 2 models trên Multi-Language
# Môi trường: Google Colab (GPU T4 hoặc A100)
# Models: mBERT | XLM-RoBERTa
# Languages: de | en | es | fr | ja | zh
# =============================================================================


# =============================================================================
# STEP 0 — Cài đặt thư viện (chạy 1 lần, restart runtime sau đó)
# =============================================================================

# !pip install -q \
#     transformers>=4.41.0 \
#     datasets==2.19.0 \
#     evaluate==0.4.2 \
#     accelerate>=0.29.3 \
#     sentencepiece==0.2.0


# =============================================================================
# STEP 1 — Import và cấu hình chung
# =============================================================================

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID   = {"negative": 0, "neutral": 1, "positive": 2}
NUM_LABELS = 3
LANGS      = ["de", "en", "es", "fr", "ja", "zh"]


# =============================================================================
# STEP 2 — Đường dẫn data và cấu hình model
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = Path("/content/drive/MyDrive/Chuyên Đề 4/Multi-language/datasets")

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH   = DATA_DIR / "val.csv"
TEST_PATH  = DATA_DIR / "test.csv"

OUT_DIR = Path("results/multi_language")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "mBERT": {
        "model_name": "bert-base-multilingual-cased",
        "max_length": 128,
        "batch_size": 32,
        "lr":         2e-5,
        "epochs":     3,
    },
    "XLM-RoBERTa": {
        "model_name": "xlm-roberta-base",
        "max_length": 128,
        "batch_size": 32,
        "lr":         2e-5,
        "epochs":     3,
    },
}

print(f"Data   : {DATA_DIR}")
print(f"Output : {OUT_DIR}")
print(f"Models : {list(MODEL_CONFIGS.keys())}")
print(f"Langs  : {LANGS}")


# =============================================================================
# STEP 3 — Load và kiểm tra data
# =============================================================================

def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = next(
        (c for c in df.columns if c.lower() in ["text", "sentence", "review", "clean_text"]),
        df.columns[0]
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ["label", "sentiment", "stars"]),
        df.columns[1]
    )
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


train_df = load_split(TRAIN_PATH)
val_df   = load_split(VAL_PATH)
test_df  = load_split(TEST_PATH)

print(f"\n{'='*50}")
print(f"Train  : {len(train_df):,} rows")
print(f"Val    : {len(val_df):,} rows")
print(f"Test   : {len(test_df):,} rows")
print(f"\nLabel distribution (train):")
print(train_df["label"].value_counts().sort_index())
if "lang" in train_df.columns:
    print(f"\nDistribution by language (train):")
    print(train_df.groupby(["lang", "label"]).size().unstack().fillna(0).astype(int))


# =============================================================================
# STEP 4 — Dataset class và DataLoader
# =============================================================================

class SentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts      = df["text"].tolist()
        self.labels     = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if "token_type_ids" in enc:
            tti = enc["token_type_ids"].squeeze(0)
        else:
            tti = torch.zeros(self.max_length, dtype=torch.long)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": tti,
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_loaders(tokenizer, cfg: dict):
    train_ds = SentimentDataset(train_df, tokenizer, cfg["max_length"])
    val_ds   = SentimentDataset(val_df,   tokenizer, cfg["max_length"])
    test_ds  = SentimentDataset(test_df,  tokenizer, cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


# =============================================================================
# STEP 5 — Train loop và Evaluate loop
# =============================================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        extra = {}
        if batch["token_type_ids"].any():
            extra["token_type_ids"] = batch["token_type_ids"].to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, **extra)
        scaler.scale(outputs.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss    += outputs.loss.item() * labels.size(0)
        total_correct += (outputs.logits.argmax(-1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, desc="Eval"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0

    for batch in tqdm(loader, desc=f"  {desc}", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        extra = {}
        if batch["token_type_ids"].any():
            extra["token_type_ids"] = batch["token_type_ids"].to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, **extra)

        total_loss    += outputs.loss.item() * labels.size(0)
        total_samples += labels.size(0)
        preds = outputs.logits.argmax(-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss":        total_loss / total_samples,
        "accuracy":    accuracy_score(all_labels, all_preds),
        "f1_macro":    f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "preds":       all_preds,
        "labels":      all_labels,
    }


# =============================================================================
# STEP 6 — Hàm train 1 model hoàn chỉnh
# =============================================================================

def train_model(model_key: str, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  MODEL : {model_key}  ({cfg['model_name']})")
    print(f"{'='*60}")

    model_out = OUT_DIR / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    train_loader, val_loader, test_loader = make_loaders(tokenizer, cfg)

    optimizer   = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler   = get_linear_schedule_with_warmup(
                      optimizer, int(0.1 * total_steps), total_steps)
    scaler      = torch.cuda.amp.GradScaler()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1     = -1.0
    best_model_path = model_out / "best_model"

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"\n  Epoch {epoch}/{cfg['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_metrics           = evaluate(model, val_loader, desc="Validation")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        print(f"  val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}"
              f"  val_f1={val_metrics['f1_macro']:.4f}  ({time.time()-t0:.1f}s)")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ Best model saved (val_f1={best_val_f1:.4f})")

    # ── Test với best model ──
    print(f"\n  Evaluating best model on test set...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(DEVICE)
    test_metrics = evaluate(best_model, test_loader, desc="Test")

    print(f"\n  ── TEST RESULTS ({model_key}) ──")
    print(f"  accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"  f1_macro   : {test_metrics['f1_macro']:.4f}")
    print(f"  f1_weighted: {test_metrics['f1_weighted']:.4f}")
    print(classification_report(test_metrics["labels"], test_metrics["preds"],
                                target_names=["negative", "neutral", "positive"], digits=4))

    result = {
        "model_key":   model_key,
        "model_name":  cfg["model_name"],
        "benchmark":   "multi_language",
        "best_val_f1": best_val_f1,
        "test":        {k: v for k, v in test_metrics.items() if k not in ["preds", "labels"]},
        "history":     history,
        "preds":       test_metrics["preds"],
        "labels":      test_metrics["labels"],
    }

    with open(model_out / "result.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k not in ["preds", "labels"]}, f, indent=2)
    np.save(model_out / "confusion_matrix.npy",
            confusion_matrix(test_metrics["labels"], test_metrics["preds"]))

    return result


# =============================================================================
# STEP 7 — Chạy benchmark 2 model
# =============================================================================

all_results = {}

for model_key, cfg in MODEL_CONFIGS.items():
    all_results[model_key] = train_model(model_key, cfg)

    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory freed after {model_key}")

print(f"\n{'='*60}")
print("  DONE — cả 2 model đã chạy xong")
print('='*60)


# =============================================================================
# STEP 8 — Bảng so sánh + biểu đồ
# =============================================================================

summary_df = pd.DataFrame([
    {
        "Model":       k,
        "Accuracy":    round(v["test"]["accuracy"],    4),
        "F1 Macro":    round(v["test"]["f1_macro"],    4),
        "F1 Weighted": round(v["test"]["f1_weighted"], 4),
        "Val F1 Best": round(v["best_val_f1"],         4),
    }
    for k, v in all_results.items()
]).sort_values("F1 Macro", ascending=False)

print(f"\n{'='*60}")
print("  SUMMARY — multi_language")
print('='*60)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

COLORS      = {"mBERT": "#534AB7", "XLM-RoBERTa": "#0F6E56"}
model_names = list(all_results.keys())
colors      = [COLORS[m] for m in model_names]

# ── Plot 1: Bar chart F1 + Accuracy ──
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Benchmark: Multi-Language — mBERT vs XLM-RoBERTa", fontsize=13, fontweight="bold")

for ax, metric, title in zip(axes,
                              ["f1_macro", "accuracy"],
                              ["F1 Macro (Test)", "Accuracy (Test)"]):
    scores = [all_results[m]["test"][metric] for m in model_names]
    bars   = ax.bar(model_names, scores, color=colors, alpha=0.85, edgecolor="white", width=0.4)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "comparison_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: comparison_bar.png")

# ── Plot 2: Training curve ──
fig, ax = plt.subplots(figsize=(8, 5))
for m, color in zip(model_names, colors):
    ax.plot(range(1, len(all_results[m]["history"]["val_f1"]) + 1),
            all_results[m]["history"]["val_f1"],
            marker="o", label=m, color=color, linewidth=2)
ax.set_title("Validation F1 Macro per Epoch — Multi-Language")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Macro")
ax.set_ylim(0, 1.0)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: training_curves.png")

# ── Plot 3: Confusion matrix ──
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
label_names = ["neg", "neu", "pos"]
for ax, m in zip(axes, model_names):
    cm = confusion_matrix(all_results[m]["labels"], all_results[m]["preds"], normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_title(f"{m}\nF1={all_results[m]['test']['f1_macro']:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

fig.suptitle("Normalized Confusion Matrix — Multi-Language", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: confusion_matrices.png")


# =============================================================================
# STEP 9 — Phân tích F1 theo từng ngôn ngữ (de/en/es/fr/ja/zh)
# =============================================================================

def analyze_per_language(results: dict, df: pd.DataFrame):
    if "lang" not in df.columns:
        print("Cột 'lang' không tồn tại, bỏ qua per-language analysis.")
        return

    rows = []
    for model_key, res in results.items():
        tmp = df.copy()
        tmp["pred"] = res["preds"]
        for lang in sorted(tmp["lang"].unique()):
            sub = tmp[tmp["lang"] == lang]
            f1  = f1_score(sub["label"], sub["pred"], average="macro", zero_division=0)
            rows.append({"Model": model_key, "lang": lang, "F1 Macro": round(f1, 4)})

    per_lang_df = pd.DataFrame(rows)
    pivot = per_lang_df.pivot(index="lang", columns="Model", values="F1 Macro")
    print("\nPer-language F1 Macro:")
    print(pivot.to_string())
    per_lang_df.to_csv(OUT_DIR / "per_language_f1.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, vmin=0.5, vmax=1.0)
    ax.set_title("F1 Macro per Language — Multi-Language")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "heatmap_per_language.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: heatmap_per_language.png")

    # ── Plot thêm: grouped bar per language ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(pivot.index))
    width = 0.35
    for i, model_key in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[model_key], width,
               label=model_key, color=COLORS[model_key], alpha=0.85)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Macro")
    ax.set_title("F1 Macro per Language — mBERT vs XLM-RoBERTa")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bar_per_language.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: bar_per_language.png")


analyze_per_language(all_results, test_df)


# =============================================================================
# STEP 10 — Lưu JSON tổng hợp
# =============================================================================

final_summary = {
    "benchmark": "multi_language",
    "timestamp": datetime.now().isoformat(),
    "languages": LANGS,
    "models": {
        k: {
            "model_name":  v["model_name"],
            "accuracy":    v["test"]["accuracy"],
            "f1_macro":    v["test"]["f1_macro"],
            "f1_weighted": v["test"]["f1_weighted"],
            "best_val_f1": v["best_val_f1"],
        }
        for k, v in all_results.items()
    },
}

with open(OUT_DIR / "final_summary.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  Tất cả kết quả lưu tại: {OUT_DIR}")
print(f"  Files:")
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"    {p.relative_to(OUT_DIR)}")
print('='*60)

# =============================================================================
# CẤU TRÚC OUTPUT
# results/multi_language/
# ├── summary.csv
# ├── final_summary.json
# ├── comparison_bar.png
# ├── training_curves.png
# ├── confusion_matrices.png
# ├── heatmap_per_language.png
# ├── bar_per_language.png
# ├── per_language_f1.csv
# ├── mBERT/
# │   ├── best_model/
# │   ├── result.json
# │   └── confusion_matrix.npy
# └── XLM-RoBERTa/
#     ├── best_model/
#     ├── result.json
#     └── confusion_matrix.npy
# =============================================================================