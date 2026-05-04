import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Sử dụng đường dẫn tương đối dựa trên vị trí của script
SCRIPT_DIR = Path(__file__).parent
BASE = SCRIPT_DIR.parent.parent  # Từ multidomain -> benchmarks_datasets -> Final_Project
OUT = SCRIPT_DIR  # Lưu vào thư mục chứa script
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("amazon_reviews",        BASE / "datasets" / "amazon_reviews_multi" / "en" / "train.csv",  "text", "label"),
    ("twitter_sentiment",     BASE / "datasets" / "twitter_sentiment_en" / "Twitter_Data_processed.csv", "text", "label"),
    ("yelp_restaurant_reviews", BASE / "datasets" / "yelp_restaurant_reviews" / "Yelp_Restaurant_Reviews_processed.csv", "text", "label"),
]

frames = []
for domain_name, path, text_col, label_col in SOURCES:
    df = pd.read_csv(path, usecols=[text_col, label_col])
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["domain"] = domain_name
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    # Sampling từng group để tránh FutureWarning
    sampled = []
    for label_val, group in df.groupby("label"):
        sampled.append(group.sample(min(len(group), 2000), random_state=42))
    df = pd.concat(sampled, ignore_index=True)
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
full = full[["text", "label", "domain"]].sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total: {len(full)} rows")
print(full.groupby(["domain", "label"]).size().unstack())

# stratified split theo domain × label
strat_col = full["domain"] + "_" + full["label"].astype(str)

train_val, test  = train_test_split(full, test_size=0.15, stratify=strat_col, random_state=42)
strat_col2       = train_val["domain"] + "_" + train_val["label"].astype(str)
train, val       = train_test_split(train_val, test_size=0.15/0.85, stratify=strat_col2, random_state=42)

train.to_csv(OUT / "train.csv", index=False)
val.to_csv(OUT  / "val.csv",   index=False)
test.to_csv(OUT / "test.csv",  index=False)

print(f"\nSplit: train={len(train)} | val={len(val)} | test={len(test)}")
print("Saved to:", OUT)