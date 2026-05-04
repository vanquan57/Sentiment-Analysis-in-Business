# Sentiment Analysis in Business — Multi-Domain & Multi-Language NLP Project

**TRẠNG THÁI: HOÀN THÀNH 100% ✅**

---

## 📋 Tóm tắt Project

Đây là một dự án **Sentiment Analysis** (phân tích cảm xúc) sử dụng các mô hình transformer đa-ngôn ngữ (**mBERT**, **XLM-RoBERTa**) trên dữ liệu từ **nhiều domain** (Amazon, Twitter, Yelp) và **nhiều ngôn ngữ** (Tiếng Anh, Tiếng Đức, Tiếng Tây Ban Nha, Tiếng Pháp, Tiếng Nhật, Tiếng Trung, Tiếng Việt).

**Các kết quả chính:**
- ✅ XLM-RoBERTa vượt trội về accuracy & F1-score
- ✅ Transfer learning cross-lingual hoạt động hiệu quả
- ✅ Domain adaptation được đo lường chi tiết
- ✅ Model cuối cùng đạt **Accuracy 77.68%** & **F1 Macro 77.61%** trên Mixed Dataset

---

## 📁 Cấu trúc Project Chi Tiết

```
Final_Project/
├── analytis.py                          # [Tool] Thống kê tất cả datasets
├── gen_tree.py                          # [Tool] Sinh ra cây thư mục project
├── requirements.txt                     # Dependencies
├── dataset_statistics_summary.csv       # Bảng thống kê tất cả datasets
├── structure.tree                       # Output cây thư mục
│
├── benchmarks_datasets/                 # 📊 Ba benchmark chính của project
│   │
│   ├── multidomain/                    # Benchmark 1: Multi-Domain (3 domains)
│   │   ├── build_multidomain_en.py     # [Build] Xây dựng dataset
│   │   ├── train_model.py              # [Train] Benchmark mBERT vs XLM-RoBERTa
│   │   ├── train.csv                   # 10,710 mẫu (train)
│   │   ├── val.csv                     # 2,295 mẫu (validation)
│   │   └── test.csv                    # 4,995 mẫu (test)
│   │
│   ├── multilanguage/                  # Benchmark 2: Multi-Language (6 languages)
│   │   ├── build_multilanguage_amazon.py    # [Build] Xây dựng dataset
│   │   ├── train_model.py                   # [Train] Benchmark mBERT vs XLM-RoBERTa
│   │   ├── train.csv                        # 25,500 mẫu (train)
│   │   ├── val.csv                         # 5,100 mẫu (validation)
│   │   └── test.csv                        # 5,400 mẫu (test)
│   │
│   └── mixed/                           # Benchmark 3: Mixed (Multi-domain + Multi-lang)
│       ├── build_mixed.py              # [Build] Xây dựng dataset
│       ├── train_model.py              # [Train] Train XLM-RoBERTa (model chọn cuối cùng)
│       ├── train.csv                   # 29,400 mẫu (train)
│       ├── val.csv                     # 6,300 mẫu (validation)
│       └── test.csv                    # 6,300 mẫu (test)
│
└── datasets/                            # 📚 Raw datasets từ nhiều nguồn
    │
    ├── amazon_reviews_multi/           # Amazon Reviews Multi-Language Dataset
    │   ├── amazon_reviews_multi.py     # [Script] Tải dataset từ HuggingFace
    │   ├── convert_to_csv.py           # [Script] Convert JSONL → CSV
    │   ├── README.md
    │   ├── de/                         # Tiếng Đức (Deutsch)
    │   │   ├── train.csv               # 6,000 mẫu
    │   │   ├── train.jsonl
    │   │   ├── validation.jsonl
    │   │   └── test.jsonl
    │   ├── en/                         # Tiếng Anh (English)
    │   ├── es/                         # Tiếng Tây Ban Nha (Español)
    │   ├── fr/                         # Tiếng Pháp (Français)
    │   ├── ja/                         # Tiếng Nhật (日本語)
    │   └── zh/                         # Tiếng Trung (中文)
    │
    ├── hung20gg_NEU_ESC/               # NEU & ESC Sentiment Dataset (Vietnamese)
    │   ├── NEU_ESC_processed.csv       # 6,000 mẫu, đã xử lý
    │   ├── NEU_ESC.process.py          # [Script] Xử lý dữ liệu
    │   └── train_set.csv
    │
    ├── twitter_sentiment_en/           # Twitter Sentiment Dataset (English)
    │   ├── Twitter_Data_processed.csv  # 6,000 mẫu, đã xử lý
    │   ├── Twitter_Data.csv            # Raw data
    │   └── process_twitter_data.py     # [Script] Xử lý dữ liệu
    │
    ├── yelp_restaurant_reviews/        # Yelp Restaurant Reviews (English)
    │   ├── Yelp_Restaurant_Reviews_processed.csv  # 6,000 mẫu, đã xử lý
    │   ├── Yelp Restaurant Reviews.csv # Raw data
    │   └── yelp_restaurant_reviews.process.py     # [Script] Xử lý dữ liệu
    │
    └── vietnamese_students_feedback/   # Vietnamese Students' Feedback Corpus
        ├── vietnamese_students_feedback_processed.csv # 6,000 mẫu, đã xử lý
        ├── vietnamese_students_feedback_process.py    # [Script] Xử lý dữ liệu
        ├── feedback_neutral_1542.csv
        ├── sents.txt
        ├── sentiments.txt
        └── README.md
```

---

## 📊 Chi Tiết Từng Thư Mục

### 1️⃣ **`benchmarks_datasets/` — Ba Benchmark Chính**

#### **1.1 multidomain/** — Benchmark Multi-Domain
**Mục đích:** So sánh mBERT vs XLM-RoBERTa trên **3 domains khác nhau** để kiểm tra **domain adaptation**.

**Dữ liệu:**
- Nguồn: Amazon EN + Twitter EN + Yelp EN
- Tổng: 18,000 mẫu (2,000 mẫu/label/domain)
- Split: Train=10,710 | Val=2,295 | Test=4,995
- Stratified split theo `domain × label`

**Files:**
- `build_multidomain_en.py` — Tạo dataset bằng cách:
  1. Load 3 csv từ 3 domains
  2. Normalize cột "text" và "label"
  3. Sample 2,000 mẫu/label/domain
  4. Stratified split (test 15%, val 15% của train)

- `train_model.py` — Chạy benchmark:
  1. Load train/val/test CSV
  2. Tokenize text với `AutoTokenizer`
  3. Fine-tune 2 models (mBERT, XLM-RoBERTa) — 3 epochs
  4. Đánh giá accuracy, F1-score per domain (domain adaptation test)
  5. Lưu kết quả & visualizations

**Kết quả Domain Adaptation (Held-out test):**
```
Model: mBERT
- Amazon: -0.208 F1 drop
- Twitter: -0.347 F1 drop (khó nhất!)
- Yelp: -0.167 F1 drop

Model: XLM-RoBERTa
- Amazon: -0.153 F1 drop
- Twitter: -0.342 F1 drop
- Yelp: -0.161 F1 drop
```

---

#### **1.2 multilanguage/** — Benchmark Multi-Language
**Mục đích:** So sánh mBERT vs XLM-RoBERTa trên **6 ngôn ngữ** để kiểm tra **cross-lingual transfer**.

**Dữ liệu:**
- Nguồn: Amazon Reviews (6 ngôn ngữ: de, en, es, fr, ja, zh)
- Tổng: 36,000 mẫu (2,000 mẫu/label/language)
- Split: Train=25,500 | Val=5,100 | Test=5,400
- Stratified split theo `language × label`

**Files:**
- `build_multilanguage_amazon.py` — Tạo dataset:
  1. Loop qua 6 ngôn ngữ, load từng train.csv
  2. Auto-detect cột "text" và "label"
  3. Sample 2,000 mẫu/label/ngôn ngữ
  4. Stratified split

- `train_model.py` — Benchmark:
  1. Train 2 models trên tất cả 6 ngôn ngữ
  2. Cross-lingual transfer test: Train on EN → Test on others
  3. Đo F1-drop cho mỗi language pair

**Kết quả Cross-Lingual Transfer (Train EN → Test Others):**
```
Model: mBERT
- de: -0.110 F1 drop
- es: -0.121 F1 drop
- fr: -0.135 F1 drop
- ja: -0.143 F1 drop
- zh: -0.219 F1 drop (rất khó!)

Model: XLM-RoBERTa
- de: -0.017 F1 drop ⭐ (gần như không drop!)
- es: -0.026 F1 drop ⭐
- fr: -0.025 F1 drop ⭐
- ja: -0.013 F1 drop ⭐
- zh: -0.051 F1 drop ⭐
```

**Insight:** XLM-RoBERTa transfer tốt hơn rất nhiều!

---

#### **1.3 mixed/** — Benchmark Mixed (Production Model)
**Mục đích:** Train **XLM-RoBERTa** (model được chọn) trên **multi-domain + multi-language** dataset lớn.

**Dữ liệu:**
- Nguồn: 
  - Amazon (3 ngôn ngữ: en, zh, ja — đại diện Latin, CJK logographic, CJK syllabic)
  - Twitter (en)
  - Yelp (en)
  - NEU_ESC (vi)
  - Vietnamese Students Feedback (vi)
- Tổng: 42,000 mẫu (2,000 mẫu/label/source)
- Split: Train=29,400 | Val=6,300 | Test=6,300
- Stratified split theo `domain × language × label`

**Files:**
- `build_mixed.py` — Tạo dataset:
  1. Load từ 7 sources khác nhau
  2. Auto-detect cột text/label với fallback logic
  3. Sample 2,000 mẫu/label/source
  4. Thống kê phân phối theo domain × lang
  5. Stratified split

- `train_model.py` — Train XLM-RoBERTa:
  1. Load mixed dataset
  2. Fine-tune XLM-RoBERTa — 3 epochs
  3. Lưu best model theo **val_f1_macro** (không phải val_loss)
  4. Đánh giá per-domain & per-language
  5. Save model → Production

**Kết quả Final Model (XLM-RoBERTa on Mixed):**
```
Overall:
- Accuracy: 0.7768 (77.68%)
- F1 Macro: 0.7761 (77.61%)
- F1 Weighted: 0.7761

Per-Domain F1:
- feedback_student: 0.9153 ⭐ (dữ liệu nhỏ nhưng mô hình tốt)
- twitter: 0.7534
- yelp: 0.7545
- amazon: 0.7524
- neu_esc: 0.7521

Per-Language F1:
- vi (Vietnamese): 0.8340 ⭐
- en (English): 0.7608
- ja (Japanese): 0.7590
- zh (Chinese): 0.7237 (khó nhất)
```

---

### 2️⃣ **`datasets/` — Raw Datasets từ Nhiều Nguồn**

#### **2.1 amazon_reviews_multi/**
**Nguồn:** [Amazon Reviews Multi Language Dataset](https://huggingface.co/datasets/amazon_reviews_multi) từ HuggingFace

**Cấu trúc:**
```
amazon_reviews_multi/
├── amazon_reviews_multi.py   # [Script] Tải dataset từ HuggingFace
├── convert_to_csv.py         # [Script] Convert JSONL → CSV
├── README.md
└── de/, en/, es/, fr/, ja/, zh/  # 6 thư mục ngôn ngữ
    ├── train.csv             # CSV đã chuyển đổi (dùng)
    ├── train.jsonl           # JSONL gốc (raw)
    ├── validation.jsonl
    └── test.jsonl
```

**Mô tả:**
- Amazon product reviews với 3 classes: 1 star (negative), 2 stars (neutral), 3+ stars (positive)
- Mỗi ngôn ngữ: ~6,000 reviews
- Cột chính: `review_body` (text), `stars` (label 1-3)

**Files có ý nghĩa:**
- `train.csv` — CSV đã chuyển đổi từ JSONL, dùng cho các build scripts
- `amazon_reviews_multi.py` — Download từ HuggingFace
  ```python
  from datasets import load_dataset
  dataset = load_dataset("amazon_reviews_multi", "en")
  ```
- `convert_to_csv.py` — Convert JSONL → CSV cho dễ xử lý

---

#### **2.2 hung20gg_NEU_ESC/**
**Nguồn:** NEU (Neutral) & ESC (Emotion Sentiment Classification) dataset cho Tiếng Việt

**Cấu trúc:**
```
hung20gg_NEU_ESC/
├── NEU_ESC_processed.csv    # 6,000 mẫu, đã xử lý ✅ (dùng)
├── NEU_ESC.process.py       # [Script] Xử lý raw data
└── train_set.csv            # Raw data
```

**Mô tả:**
- Vietnamese sentiment dataset
- 3 classes: 0 (negative), 1 (neutral), 2 (positive)
- Mẫu câu: "Chất lượng tuyệt vời!" → positive

**Files:**
- `NEU_ESC_processed.csv` — Dataset đã xử lý:
  - Cột: `text`, `label`
  - Size: 6,000 rows
- `NEU_ESC.process.py` — Xử lý từ raw data

---

#### **2.3 twitter_sentiment_en/**
**Nguồn:** Twitter Sentiment Dataset (English)

**Cấu trúc:**
```
twitter_sentiment_en/
├── Twitter_Data_processed.csv   # 6,000 mẫu, đã xử lý ✅ (dùng)
├── Twitter_Data.csv             # Raw data
└── process_twitter_data.py      # [Script] Xử lý
```

**Mô tả:**
- English tweets với sentiment labels
- 3 classes: 0 (negative), 1 (neutral), 2 (positive)
- Đặc điểm: Text ngắn, nhiều #hashtag, @mentions, emoticons

**Files:**
- `Twitter_Data_processed.csv` — Dataset xử lý:
  - Cột: `text`, `label`
  - Size: 6,000 rows

---

#### **2.4 yelp_restaurant_reviews/**
**Nguồn:** Yelp Restaurant Reviews

**Cấu trúc:**
```
yelp_restaurant_reviews/
├── Yelp_Restaurant_Reviews_processed.csv   # 6,000 mẫu ✅ (dùng)
├── Yelp Restaurant Reviews.csv             # Raw data
└── yelp_restaurant_reviews.process.py      # [Script] Xử lý
```

**Mô tả:**
- Reviews nhà hàng với star rating
- 3 classes: 1-2 stars (negative), 3 stars (neutral), 4-5 stars (positive)
- Đặc điểm: Text dài (avg 101 words), formal, detailed descriptions

**Files:**
- `Yelp_Restaurant_Reviews_processed.csv`:
  - Cột: `text`, `label`
  - Size: 6,000 rows

---

#### **2.5 vietnamese_students_feedback/**
**Nguồn:** Vietnamese Students' Feedback Corpus

**Cấu trúc:**
```
vietnamese_students_feedback/
├── vietnamese_students_feedback_processed.csv   # 6,000 mẫu ✅ (dùng)
├── vietnamese_students_feedback_process.py      # [Script] Xử lý
├── feedback_neutral_1542.csv                    # Subset dữ liệu
├── sents.txt
├── sentiments.txt
└── README.md
```

**Mô tả:**
- Vietnamese student feedback (feedback từ sinh viên)
- 3 classes: 0 (negative), 1 (neutral), 2 (positive)
- Mẫu: "Bài học rất hay!" → positive

**Files:**
- `vietnamese_students_feedback_processed.csv`:
  - Cột: `text`, `label`
  - Size: 6,000 rows

---

### 3️⃣ **Root-level Files (Project Config)**

| File | Mục đích |
|------|---------|
| `analytis.py` | 🔧 Tool thống kê tất cả datasets — in ra bảng tổng hợp rows, labels, word count, character count. Chạy: `python analytis.py` |
| `gen_tree.py` | 🔧 Tool sinh ra cây thư mục dự án (ASCII format). Chạy: `python gen_tree.py` |
| `requirements.txt` | 📋 Dependencies chính: `datasets`, `google-generativeai`, `scikit-learn` (+ `transformers`, `torch` cần cài riêng) |
| `dataset_statistics_summary.csv` | 📊 Output CSV từ `analytis.py` — bảng tổng hợp thống kê từng dataset |
| `structure.tree` | 📁 Output cây thư mục từ `gen_tree.py` |
| `README.md` | 📖 File này |

---

## 📈 Kết Quả Benchmark Chi Tiết

### **Benchmark 1: Multi-Domain — Standard Fine-tuning**

| Model | Accuracy | F1 Macro | Nhận xét |
|-------|----------|----------|---------|
| mBERT | Thấp hơn | Thấp hơn | ❌ |
| XLM-RoBERTa | Cao hơn | Cao hơn | ✅ |

### **Benchmark 1: Multi-Domain — Domain Adaptation (Held-out Domain)**
Test unseen domains — huấn luyện trên 2 domains, test trên 1 domain khác:

```
Nguồn Training: Amazon + Twitter → Test: Yelp
Nguồn Training: Amazon + Yelp → Test: Twitter
Nguồn Training: Twitter + Yelp → Test: Amazon
```

| Held-out | mBERT F1 Delta | XLM-RoBERTa F1 Delta |
|----------|----------------|---------------------|
| amazon | -0.208 | -0.153 |
| twitter | -0.347 ⚠️ | -0.342 ⚠️ |
| yelp | -0.167 | -0.161 |

**Insight:** Twitter khó nhất cho cả 2 model (short, noisy text).

---

### **Benchmark 2: Multi-Language — Standard Fine-tuning**

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| mBERT | 0.7219 | 0.7210 |
| XLM-RoBERTa | 0.7548 ✅ | 0.7530 ✅ |

---

### **Benchmark 2: Multi-Language — Cross-Lingual Transfer**
Huấn luyện trên EN → Test trên các ngôn ngữ khác:

| Target Language | mBERT F1 Delta | XLM-RoBERTa F1 Delta |
|-----------------|----------------|---------------------|
| de (Deutsch) | -0.110 | -0.017 ✅ |
| es (Español) | -0.121 | -0.026 ✅ |
| fr (Français) | -0.135 | -0.025 ✅ |
| ja (日本語) | -0.143 | -0.013 ✅ |
| zh (中文) | -0.219 ⚠️ | -0.051 ✅ |

**Insight:** XLM-RoBERTa gần như không mất F1 khi transfer. mBERT drop mạnh ở Chinese.

---

### **Benchmark 3: Mixed — Final Model (XLM-RoBERTa)**

**Overall Performance:**
```
Accuracy:      0.7768 (77.68%)
F1 Macro:      0.7761 (77.61%)
F1 Weighted:   0.7761
```

**Per-Domain Performance:**
```
Feedback Student: F1 = 0.9153 ⭐ (smallest but best)
Twitter:          F1 = 0.7534
Yelp:             F1 = 0.7545
Amazon:           F1 = 0.7524
NEU_ESC:          F1 = 0.7521
```

**Per-Language Performance:**
```
Vietnamese:  F1 = 0.8340 ⭐ (surprising good!)
English:     F1 = 0.7608
Japanese:    F1 = 0.7590
Chinese:     F1 = 0.7237 (hardest)
```

---

## 🎓 Kết Luận Khoa Học

### ✅ Kết Quả Chính:

1. **XLM-RoBERTa thắng toàn diện:**
   - Standard fine-tuning: Accuracy +3.3% vs mBERT
   - Cross-lingual transfer: F1 drop avg -0.026 vs mBERT -0.125
   - Domain adaptation: Tương đương nhưng ổn định hơn

2. **mBERT yếu ở cross-lingual:**
   - Chinese: -21.9% F1 drop (máy tính CJK khó)
   - Lý do: Tokenization không có whitespace → token explosion

3. **Domains khó nhất: Twitter**
   - Cả 2 model drop -0.34 F1 khi unseen
   - Lý do: Short, noisy, informal text

4. **Vietnamese transfer tốt:**
   - F1 = 0.8340 trên mixed dataset
   - Vượt trội so với 0.7608 (English)
   - Lý do: Underfitting → mô hình không overfitting

5. **Amazon chiếm 43% Mixed Dataset:**
   - Limitation: Kết quả có thể skew về Amazon domain
   - Recommend: Balanced sampling hoặc weighted loss

---

## 🚀 Hướng Dùng & Reproducing

### 1️⃣ **Setup Environment**
```bash
# Clone project
git clone <repo>
cd Final_Project

# Create virtual environment
python -m venv venv310
source venv310/Scripts/activate  # Windows: .\venv310\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install transformers torch sentencepiece accelerate evaluate

# For GPU (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2️⃣ **Analyze Datasets**
```bash
# Thống kê tất cả datasets
python analytis.py

# Output: dataset_statistics_summary.csv
```

### 3️⃣ **Rebuild Benchmarks (Optional)**

#### Multidomain:
```bash
cd benchmarks_datasets/multidomain
python build_multidomain_en.py    # Tạo train/val/test CSV
# Copy lên Google Drive hoặc Colab
python train_model.py              # Chạy benchmark (cần Colab + GPU)
```

#### Multilanguage:
```bash
cd benchmarks_datasets/multilanguage
python build_multilanguage_amazon.py
python train_model.py
```

#### Mixed:
```bash
cd benchmarks_datasets/mixed
python build_mixed.py
python train_model.py
```

### 4️⃣ **Production Model**

Model XLM-RoBERTa fine-tuned trên Mixed dataset:
```
Saved Path: MyDrive/Chuyên Đề 4/Mixed/saved_model/xlm-roberta-mixed/
Files:
  - pytorch_model.bin
  - config.json
  - tokenizer.json
  - etc.
```

**Sử dụng model:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "MyDrive/Chuyên Đề 4/Mixed/saved_model/xlm-roberta-mixed"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Inference
text = "Sản phẩm này tuyệt vời!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()

print(f"Prediction: {pred}")  # 0=negative, 1=neutral, 2=positive
```

### 5️⃣ **Flask API (Production Deployment)**

```bash
# Download model từ Google Drive → local folder
cp -r "saved_model/xlm-roberta-mixed" ./

# Create app.py
# (app.py saved cùng thư mục model)

# Install Flask
pip install flask

# Run API
python app.py
```

**API Endpoints:**
```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sản phẩm này rất tốt!"}'

# Batch prediction
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

---

## 🔧 Lưu Ý Kỹ Thuật Quan Trọng

### 1. **Xử lý Token Type IDs**
- **mBERT:** Có token_type_ids → cần tokenizer.return_token_type_ids=True
- **XLM-RoBERTa:** KHÔNG có token_type_ids
  ```python
  inputs = tokenizer(text, return_tensors="pt")
  # XLM-RoBERTa sẽ không có "token_type_ids"
  extra = {}
  if "token_type_ids" in inputs:
      extra["token_type_ids"] = inputs["token_type_ids"]
  outputs = model(**inputs, **extra)
  ```

### 2. **num_workers=0 (Colab Warning)**
- Để tắt warning DataLoader trên Colab:
  ```python
  DataLoader(dataset, num_workers=0, ...)
  ```

### 3. **Stratified Split**
- Split theo domain × language × label để ensure representativeness:
  ```python
  strat_col = df["domain"] + "_" + df["lang"] + "_" + df["label"].astype(str)
  train_val, test = train_test_split(df, test_size=0.15, stratify=strat_col)
  ```

### 4. **Save Best Model by F1, Not Loss**
- Validation loss không đảm bảo best model
- Save theo val_f1_macro instead:
  ```python
  if val_f1_macro > best_f1:
      best_f1 = val_f1_macro
      model.save_pretrained(save_path)
  ```

### 5. **Auto-detect Columns**
- Datasets không chuẩn → auto-detect text/label columns:
  ```python
  text_col = next(c for c in df.columns if "text" in c.lower())
  label_col = next(c for c in df.columns if "label" in c.lower())
  ```

### 6. **Limitation: Amazon Dominates**
- Mixed dataset: Amazon = 43% (18,000 / 42,000)
- Recommendation cho thesis: Mention trong limitation section

---

## 📚 File Details Summary

### **Build Scripts** (Tạo dataset từ raw sources)
```python
# File: benchmarks_datasets/*/build_*.py
# Cách chạy: python build_*.py
# Output: train.csv, val.csv, test.csv

Bước:
1. Load từ raw datasets
2. Normalize columns (text, label)
3. Sample balanced theo labels
4. Stratified split (test 15%, val 15% of train, train còn lại)
5. Save to CSV
```

### **Train Scripts** (Benchmark models)
```python
# File: benchmarks_datasets/*/train_model.py
# Cách chạy: Colab notebook (GPU required)
# Output: results/, trained models, visualizations

Bước:
1. Load train/val/test CSV
2. Tokenize + create DataLoader
3. Fine-tune models (AdamW optimizer, linear warmup)
4. Evaluate per epoch (accuracy, F1-score)
5. Save best model + generate reports
```

### **Analysis Scripts** (Thống kê)
```python
# File: analytis.py
# Cách chạy: python analytis.py
# Output: dataset_statistics_summary.csv + console output

Chỉ số:
- rows_read, rows_skipped
- Per-label distribution
- Mean/median characters
- Mean/median words
```

---

## 📖 Dataset Schema

Tất cả CSV đều có schema chuẩn:

### **Multidomain / Multilanguage / Mixed CSV:**
```csv
text,label,domain[,lang]
"Text content here",0,amazon[,en]
"Excellent product!",2,twitter[,en]
...
```

**Cột:**
- `text` (str): Input text
- `label` (int): 0=negative, 1=neutral, 2=positive
- `domain` (str): amazon | twitter | yelp | neu_esc | feedback_student
- `lang` (str, optional): en | de | es | fr | ja | zh | vi

---

## 🎯 Mục Tiêu & Thành Tựu

✅ **Hoàn thành:**
- Multi-domain benchmark (3 domains × 2 models)
- Multi-language benchmark (6 languages × 2 models)
- Mixed dataset (5 sources × 7 languages × 3 labels)
- Production model (XLM-RoBERTa trên mixed)
- Cross-lingual transfer analysis
- Domain adaptation analysis
- Flask API for deployment

❌ **Ngoài scope:**
- Deploy trên cloud (AWS/Azure) — local Flask API đủ
- Fine-tune XLM-RoBERTa lớn hơn (roberta-large) — base sufficient
- Non-English evaluation (chỉ focus 3 classes)

---

## 🤝 Dependencies

```
Core:
- transformers >= 4.41.0
- torch >= 2.0.0
- scikit-learn >= 1.3.0
- pandas >= 1.5.0
- numpy >= 1.23.0

Optional (for Colab):
- google-generativeai
- datasets == 2.19.0
- evaluate == 0.4.2
- accelerate >= 0.29.3
- sentencepiece == 0.2.0

Web API:
- flask >= 2.3.0
```

---

## 📞 Contact & Support

**Project Status:** ✅ Completed 100%

Nếu có câu hỏi hoặc muốn extend project, hãy liên hệ hoặc tạo issue/PR.

---

*Last updated: 2024 | Sentiment Analysis in Business — Final Project*
