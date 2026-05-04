import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Sử dụng đường dẫn tương đối dựa trên vị trí của script
SCRIPT_DIR = Path(__file__).parent
BASE = SCRIPT_DIR.parent.parent  # Từ multilanguage -> benchmarks_datasets -> Final_Project
OUT = SCRIPT_DIR  # Lưu vào thư mục chứa script
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["de", "en", "es", "fr", "ja", "zh"]

frames = []
for lang in LANGS:
    path = BASE / "datasets" / "amazon_reviews_multi" / lang / "train.csv"
    df = pd.read_csv(path)

    # tự động detect cột text và label
    text_col  = next(c for c in df.columns if "text" in c.lower() or "review" in c.lower())
    label_col = next(c for c in df.columns if "label" in c.lower() or "star" in c.lower() or "sentiment" in c.lower())

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["lang"] = lang
    df["domain"] = "amazon_reviews"
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    # Sampling từng group để tránh FutureWarning
    sampled = []
    for label_val, group in df.groupby("label"):
        sampled.append(group.sample(min(len(group), 2000), random_state=42))
    df = pd.concat(sampled, ignore_index=True)
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
full = full[["text", "label", "lang", "domain"]].sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total: {len(full)} rows")
print(full.groupby(["lang", "label"]).size().unstack())

strat_col      = full["lang"] + "_" + full["label"].astype(str)
train_val, test = train_test_split(full, test_size=0.15, stratify=strat_col, random_state=42)
strat_col2      = train_val["lang"] + "_" + train_val["label"].astype(str)
train, val      = train_test_split(train_val, test_size=0.15/0.85, stratify=strat_col2, random_state=42)

train.to_csv(OUT / "train.csv", index=False)
val.to_csv(OUT  / "val.csv",   index=False)
test.to_csv(OUT / "test.csv",  index=False)

print(f"\nSplit: train={len(train)} | val={len(val)} | test={len(test)}")
print("Saved to:", OUT)