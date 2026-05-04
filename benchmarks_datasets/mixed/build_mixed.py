# build_mixed.py
# Lấy thẳng từ raw/processed files, không dùng lại data của multidomain/multilanguage

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent
BASE       = SCRIPT_DIR.parent.parent   # mixed -> benchmarks_datasets -> Final_Project
OUT        = SCRIPT_DIR
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_PER_GROUP = 2000  # mỗi (domain × lang × label), chỉnh tùy ý

# ──────────────────────────────────────────────
# Nguồn: (path, domain, lang)
# Amazon: chỉ giữ en, zh, ja để cân bằng domain
#   en — Latin, high-resource
#   zh — CJK logographic
#   ja — CJK syllabic
# ──────────────────────────────────────────────
SOURCES = [
    # Amazon — 3 lang đại diện 3 script
    (BASE / "datasets" / "amazon_reviews_multi" / "en" / "train.csv", "amazon", "en"),
    (BASE / "datasets" / "amazon_reviews_multi" / "zh" / "train.csv", "amazon", "zh"),
    (BASE / "datasets" / "amazon_reviews_multi" / "ja" / "train.csv", "amazon", "ja"),

    # Twitter EN — processed
    (BASE / "datasets" / "twitter_sentiment_en" / "Twitter_Data_processed.csv", "twitter", "en"),

    # Yelp EN — processed
    (BASE / "datasets" / "yelp_restaurant_reviews" / "Yelp_Restaurant_Reviews_processed.csv", "yelp", "en"),

    # NEU_ESC VI — processed
    (BASE / "datasets" / "hung20gg_NEU_ESC" / "NEU_ESC_processed.csv", "neu_esc", "vi"),

    # Vietnamese students feedback VI — processed
    (BASE / "datasets" / "vietnamese_students_feedback" / "vietnamese_students_feedback_processed.csv", "feedback_student", "vi"),
]


# ──────────────────────────────────────────────
# Load + auto-detect cột + sample
# ──────────────────────────────────────────────
frames = []

for path, domain, lang in SOURCES:
    if not path.exists():
        print(f"[SKIP] Không tìm thấy: {path}")
        continue

    df = pd.read_csv(path)

    # auto-detect text col
    text_col = next(
        (c for c in df.columns if c.lower() in ["text", "review_body", "review", "sentence", "clean_text"]),
        next((c for c in df.columns if "text" in c.lower() or "review" in c.lower()), df.columns[0])
    )
    # auto-detect label col
    label_col = next(
        (c for c in df.columns if c.lower() in ["label", "sentiment", "stars", "star"]),
        next((c for c in df.columns if "label" in c.lower() or "star" in c.lower()), df.columns[1])
    )

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["domain"] = domain
    df["lang"]   = lang
    df = df[["text", "label", "domain", "lang"]].dropna()
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0]

    # sample cân bằng theo label
    sampled = []
    for label_val, group in df.groupby("label"):
        sampled.append(group.sample(min(len(group), SAMPLE_PER_GROUP), random_state=42))
    df = pd.concat(sampled, ignore_index=True)

    frames.append(df)
    print(f"[OK] {domain:20s} | lang={lang} | {len(df):,} rows")


# ──────────────────────────────────────────────
# Gộp + thống kê
# ──────────────────────────────────────────────
full = pd.concat(frames, ignore_index=True)
full = full.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTotal: {len(full):,} rows")
print("\nPhân phối theo domain × label:")
print(full.groupby(["domain", "label"]).size().unstack().fillna(0).astype(int))
print("\nPhân phối theo lang × label:")
print(full.groupby(["lang", "label"]).size().unstack().fillna(0).astype(int))


# ──────────────────────────────────────────────
# Stratified split theo domain × lang × label
# ──────────────────────────────────────────────
strat_col  = full["domain"] + "_" + full["lang"] + "_" + full["label"].astype(str)

train_val, test = train_test_split(full, test_size=0.15,        stratify=strat_col,  random_state=42)
strat_col2      = train_val["domain"] + "_" + train_val["lang"] + "_" + train_val["label"].astype(str)
train, val      = train_test_split(train_val, test_size=0.15/0.85, stratify=strat_col2, random_state=42)

train.to_csv(OUT / "train.csv", index=False)
val.to_csv(OUT   / "val.csv",   index=False)
test.to_csv(OUT  / "test.csv",  index=False)

print(f"\nSplit: train={len(train):,} | val={len(val):,} | test={len(test):,}")
print("Saved to:", OUT)