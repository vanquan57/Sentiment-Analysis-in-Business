# =============================================================================
# train_model.py — Train XLM-RoBERTa trên Mixed Dataset
# Môi trường: Google Colab (GPU T4 hoặc A100)
# Mixed = Multi-domain + Multi-language
# Languages: en | zh | ja | vi
# Domains  : amazon | twitter | yelp | neu_esc | feedback_student
# =============================================================================

# =============================================================================
# STEP 0 — Cài đặt thư viện
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
LANGS      = ["en", "zh", "ja", "vi"]
DOMAINS    = ["amazon", "twitter", "yelp", "neu_esc", "feedback_student"]


# =============================================================================
# STEP 2 — Đường dẫn data và cấu hình model
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = Path("/content/drive/MyDrive/Chuyên Đề 4/Mixed/datasets")

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH   = DATA_DIR / "val.csv"
TEST_PATH  = DATA_DIR / "test.csv"

OUT_DIR = Path("results/mixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Chỉ train XLM-RoBERTa — model đã được chọn từ benchmark ──
MODEL_CONFIGS = {
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
print(f"Model  : XLM-RoBERTa (selected from benchmark)")
print(f"Langs  : {LANGS}")
print(f"Domains: {DOMAINS}")


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
if "domain" in train_df.columns:
    print(f"\nDistribution by domain × lang (train):")
    print(train_df.groupby(["domain", "lang", "label"]).size().unstack().fillna(0).astype(int))


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

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)

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
# STEP 6 — Train XLM-RoBERTa trên Mixed Dataset
# =============================================================================

def train_model(model_key: str, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  MODEL : {model_key}  ({cfg['model_name']})")
    print(f"  DATA  : Mixed (multi-domain + multi-language)")
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
        "benchmark":   "mixed",
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
# STEP 7 — Chạy train
# =============================================================================

all_results = {}

for model_key, cfg in MODEL_CONFIGS.items():
    all_results[model_key] = train_model(model_key, cfg)
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("  DONE — XLM-RoBERTa trained on Mixed dataset")
print('='*60)


# =============================================================================
# STEP 8 — Kết quả tổng hợp + biểu đồ
# =============================================================================

COLORS = {"XLM-RoBERTa": "#0F6E56"}

# ── Training curve ──
fig, ax = plt.subplots(figsize=(8, 5))
for m, color in zip(all_results.keys(), COLORS.values()):
    ax.plot(range(1, len(all_results[m]["history"]["val_f1"]) + 1),
            all_results[m]["history"]["val_f1"],
            marker="o", label=m, color=color, linewidth=2)
ax.set_title("Validation F1 Macro per Epoch — Mixed Dataset")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Macro")
ax.set_ylim(0, 1.0)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: training_curves.png")

# ── Confusion matrix ──
fig, ax = plt.subplots(figsize=(5, 4))
label_names = ["neg", "neu", "pos"]
m  = "XLM-RoBERTa"
cm = confusion_matrix(all_results[m]["labels"], all_results[m]["preds"], normalize="true")
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=ax, cbar=False, linewidths=0.5)
ax.set_title(f"Confusion Matrix — Mixed Dataset\nF1={all_results[m]['test']['f1_macro']:.4f}")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: confusion_matrix.png")


# =============================================================================
# STEP 9 — Phân tích F1 theo domain và language
# =============================================================================

def analyze_per_domain_lang(result: dict, df: pd.DataFrame):
    tmp = df.copy()
    tmp["pred"] = result["preds"]

    # ── Per domain ──
    if "domain" in df.columns:
        rows = []
        for domain in sorted(tmp["domain"].unique()):
            sub = tmp[tmp["domain"] == domain]
            f1  = f1_score(sub["label"], sub["pred"], average="macro", zero_division=0)
            rows.append({"Domain": domain, "F1 Macro": round(f1, 4), "N": len(sub)})
        domain_df = pd.DataFrame(rows)
        print("\nPer-domain F1 Macro (Mixed):")
        print(domain_df.to_string(index=False))
        domain_df.to_csv(OUT_DIR / "per_domain_f1.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(domain_df["Domain"], domain_df["F1 Macro"],
                      color="#0F6E56", alpha=0.85)
        for bar, val in zip(bars, domain_df["F1 Macro"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title("F1 Macro per Domain — Mixed Dataset (XLM-RoBERTa)")
        ax.set_ylabel("F1 Macro")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "per_domain_f1.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: per_domain_f1.png")

    # ── Per language ──
    if "lang" in df.columns:
        rows = []
        for lang in sorted(tmp["lang"].unique()):
            sub = tmp[tmp["lang"] == lang]
            f1  = f1_score(sub["label"], sub["pred"], average="macro", zero_division=0)
            rows.append({"Lang": lang, "F1 Macro": round(f1, 4), "N": len(sub)})
        lang_df = pd.DataFrame(rows)
        print("\nPer-language F1 Macro (Mixed):")
        print(lang_df.to_string(index=False))
        lang_df.to_csv(OUT_DIR / "per_lang_f1.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(lang_df["Lang"], lang_df["F1 Macro"],
                      color="#0F6E56", alpha=0.85)
        for bar, val in zip(bars, lang_df["F1 Macro"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title("F1 Macro per Language — Mixed Dataset (XLM-RoBERTa)")
        ax.set_ylabel("F1 Macro")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "per_lang_f1.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: per_lang_f1.png")


analyze_per_domain_lang(all_results["XLM-RoBERTa"], test_df)


# =============================================================================
# STEP 10 — Lưu JSON tổng hợp
# =============================================================================

final_summary = {
    "benchmark":  "mixed",
    "timestamp":  datetime.now().isoformat(),
    "model":      "xlm-roberta-base",
    "languages":  LANGS,
    "domains":    DOMAINS,
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
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"    {p.relative_to(OUT_DIR)}")
print('='*60)

# =============================================================================
# CẤU TRÚC OUTPUT
# results/mixed/
# ├── final_summary.json
# ├── training_curves.png
# ├── confusion_matrix.png
# ├── per_domain_f1.png
# ├── per_domain_f1.csv
# ├── per_lang_f1.png
# ├── per_lang_f1.csv
# └── XLM-RoBERTa/
#     ├── best_model/
#     ├── result.json
#     └── confusion_matrix.npy
# =============================================================================

# =============================================================================
# STEP 11 — Demo inference + Save model cho Flask API
# Input : câu văn bản bất kỳ
# Output: nhãn sentiment + confidence score
# =============================================================================

# ── 11.1 Save model ra Drive để dùng lại ──
SAVE_DIR = Path("/content/drive/MyDrive/Chuyên Đề 4/Mixed/saved_model/xlm-roberta-mixed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

best_model_path = OUT_DIR / "XLM-RoBERTa" / "best_model"
import shutil
shutil.copytree(best_model_path, SAVE_DIR, dirs_exist_ok=True)
print(f"Model saved to Drive: {SAVE_DIR}")


# ── 11.2 Inference function ──
def load_inference_model(model_path: Path):
    """Load model + tokenizer từ path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict(text: str, tokenizer, model, max_length: int = 128) -> dict:
    """
    Predict sentiment của 1 câu.
    Returns:
        {
            "text"      : câu input,
            "label"     : "negative" | "neutral" | "positive",
            "label_id"  : 0 | 1 | 2,
            "confidence": float,
            "scores"    : {"negative": float, "neutral": float, "positive": float}
        }
    """
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    outputs    = model(input_ids=input_ids, attention_mask=attention_mask)
    probs      = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
    label_id   = int(probs.argmax())

    return {
        "text":       text,
        "label":      ID2LABEL[label_id],
        "label_id":   label_id,
        "confidence": round(float(probs[label_id]), 4),
        "scores": {
            "negative": round(float(probs[0]), 4),
            "neutral":  round(float(probs[1]), 4),
            "positive": round(float(probs[2]), 4),
        }
    }


def predict_batch(texts: list, tokenizer, model, max_length: int = 128) -> list:
    """Predict nhiều câu cùng lúc."""
    return [predict(t, tokenizer, model, max_length) for t in texts]


# ── 11.3 Load và demo ──
print("\nLoading model for inference...")
inf_tokenizer, inf_model = load_inference_model(best_model_path)

# Demo các câu thử nghiệm
DEMO_TEXTS = [
    # English
    "This product is absolutely amazing, I love it!",
    "The quality is okay, nothing special.",
    "Terrible experience, I want my money back.",
    # Vietnamese
    "Sản phẩm rất tốt, tôi rất hài lòng!",
    "Bình thường, không có gì đặc biệt.",
    "Dịch vụ tệ quá, tôi rất thất vọng.",
    # Chinese
    "这个产品非常好，我很喜欢！",
    "质量一般，没什么特别的。",
    # Japanese
    "とても良い商品です、大好きです！",
    "普通です、特別なことはありません。",
]

print(f"\n{'='*60}")
print("  DEMO INFERENCE")
print(f"{'='*60}")
for text in DEMO_TEXTS:
    result = predict(text, inf_tokenizer, inf_model)
    print(f"\n  Text      : {result['text']}")
    print(f"  Label     : {result['label'].upper()} (confidence={result['confidence']:.4f})")
    print(f"  Scores    : neg={result['scores']['negative']:.4f} | "
          f"neu={result['scores']['neutral']:.4f} | "
          f"pos={result['scores']['positive']:.4f}")


# =============================================================================
# STEP 12 — Flask API template (lưu ra file .py)
# =============================================================================

FLASK_API_CODE = '''
# app.py — Flask API cho Sentiment Analysis
# Cài đặt: pip install flask transformers torch sentencepiece
# Chạy   : python app.py
# Test   : curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\\"text\\": \\"I love this!\\"}"

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# ── Config ──
MODEL_PATH = "xlm-roberta-mixed"   # thư mục chứa model đã save
MAX_LENGTH = 128
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model lúc khởi động ──
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")


def predict(text: str) -> dict:
    enc = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs    = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
    label_id = int(probs.argmax())

    return {
        "text":       text,
        "label":      ID2LABEL[label_id],
        "label_id":   label_id,
        "confidence": round(float(probs[label_id]), 4),
        "scores": {
            "negative": round(float(probs[0]), 4),
            "neutral":  round(float(probs[1]), 4),
            "positive": round(float(probs[2]), 4),
        }
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing field: text"}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Text is empty"}), 400

    result = predict(text)
    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch_endpoint():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing field: texts"}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "texts must be a non-empty list"}), 400

    results = [predict(str(t).strip()) for t in texts]
    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
'''

# # ── Lưu app.py ra Drive ──
# flask_path = Path("/content/drive/MyDrive/Chuyên Đề 4/Mixed/saved_model/app.py")
# with open(flask_path, "w", encoding="utf-8") as f:
#     f.write(FLASK_API_CODE.strip())
# print(f"\nFlask API saved: {flask_path}")

# print(f"""
# {'='*60}
#   HƯỚNG DẪN DEPLOY FLASK API
# {'='*60}
# 1. Download 2 thứ từ Drive về máy local:
#    - thư mục: xlm-roberta-mixed/
#    - file   : app.py

# 2. Cài thư viện:
#    pip install flask transformers torch sentencepiece

# 3. Chạy API:
#    python app.py

# 4. Test API:
#    # Single prediction
#    curl -X POST http://localhost:5000/predict \\
#         -H "Content-Type: application/json" \\
#         -d '{{"text": "This product is amazing!"}}'

#    # Batch prediction  
#    curl -X POST http://localhost:5000/predict_batch \\
#         -H "Content-Type: application/json" \\
#         -d '{{"texts": ["I love it!", "Terrible!", "Sản phẩm tốt"]}}'

# 5. Response format:
#    {{
#      "text"      : "This product is amazing!",
#      "label"     : "positive",
#      "label_id"  : 2,
#      "confidence": 0.9234,
#      "scores"    : {{"negative": 0.02, "neutral": 0.05, "positive": 0.93}}
#    }}
# {'='*60}
# """)