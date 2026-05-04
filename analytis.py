"""
Thống kê tập dữ liệu đa-domain, đa-ngôn ngữ cho bài sentiment 3 nhãn (0/1/2).

Chạy: python analytis.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent

LABEL_NAMES = ("negative (0)", "neutral (1)", "positive (2)")


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    language: str
    path: Path


def all_dataset_specs() -> list[DatasetSpec]:
    amazon = ROOT / "datasets" / "amazon_reviews_multi"
    specs: list[DatasetSpec] = []
    for lang in ("de", "en", "es", "fr", "ja", "zh"):
        specs.append(
            DatasetSpec(
                domain="amazon_reviews",
                language=lang,
                path=amazon / lang / "train.csv",
            )
        )
    specs.extend(
        [
            DatasetSpec(
                domain="neu_esc",
                language="vi",
                path=ROOT / "datasets" / "hung20gg_NEU_ESC" / "NEU_ESC_processed.csv",
            ),
            DatasetSpec(
                domain="twitter_sentiment",
                language="en",
                path=ROOT
                / "datasets"
                / "twitter_sentiment_en"
                / "Twitter_Data_processed.csv",
            ),
            DatasetSpec(
                domain="vietnamese_students_feedback",
                language="vi",
                path=ROOT
                / "datasets"
                / "vietnamese_students_feedback"
                / "vietnamese_students_feedback_processed.csv",
            ),
            DatasetSpec(
                domain="yelp_restaurant_reviews",
                language="en",
                path=ROOT
                / "datasets"
                / "yelp_restaurant_reviews"
                / "Yelp_Restaurant_Reviews_processed.csv",
            ),
        ]
    )
    return specs


def pick_text_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    norm = {h.strip().lower(): h for h in fieldnames}
    for key in ("text", "clean_text"):
        if key in norm:
            return norm[key]
    return None


def pick_label_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    norm = {h.strip().lower(): h for h in fieldnames}
    if "label" in norm:
        return norm["label"]
    return None


def analyze_csv(spec: DatasetSpec) -> dict | None:
    path = spec.path
    if not path.is_file():
        return None

    text_key: str | None = None
    label_key: str | None = None
    lang_key: str | None = None

    rows_total = 0
    skipped = 0
    lang_in_file: Counter[str] = Counter()
    labels: Counter[int] = Counter()
    char_lens: list[int] = []
    word_lens: list[int] = []

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames
        text_key = pick_text_column(list(fn) if fn else [])
        label_key = pick_label_column(list(fn) if fn else [])
        if fn:
            lk = {h.strip().lower(): h for h in fn}
            if "language" in lk:
                lang_key = lk["language"]

        if not text_key or not label_key:
            return {
                "error": "Thiếu cột text/clean_text hoặc label",
                "path": path,
            }

        for row in reader:
            rows_total += 1
            raw_text = row.get(text_key)
            raw_lab = row.get(label_key)
            if raw_text is None or raw_lab is None or str(raw_text).strip() == "":
                skipped += 1
                continue
            try:
                lab = int(float(str(raw_lab).strip()))
            except (TypeError, ValueError):
                skipped += 1
                continue

            labels[lab] += 1
            t = str(raw_text)
            char_lens.append(len(t))
            word_lens.append(len(t.split()))

            if lang_key:
                lv = row.get(lang_key)
                if lv is not None and str(lv).strip():
                    lang_in_file[str(lv).strip()] += 1

    langs_report = dict(lang_in_file) if lang_in_file else [spec.language]

    return {
        "domain": spec.domain,
        "language": spec.language,
        "path": path,
        "rows_read": rows_total,
        "rows_used": rows_total - skipped,
        "skipped": skipped,
        "labels": labels,
        "langs_in_file": langs_report,
        "mean_chars": statistics.fmean(char_lens) if char_lens else 0.0,
        "median_chars": statistics.median(char_lens) if char_lens else 0.0,
        "mean_words": statistics.fmean(word_lens) if word_lens else 0.0,
        "median_words": statistics.median(word_lens) if word_lens else 0.0,
    }


def pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "—"
    return f"{100.0 * part / whole:.1f}%"


def print_separator(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}", flush=True)


def main() -> None:
    specs = all_dataset_specs()
    results: list[dict] = []
    missing: list[Path] = []

    for spec in specs:
        r = analyze_csv(spec)
        if r is None:
            missing.append(spec.path)
            continue
        if "error" in r:
            missing.append(spec.path)
            results.append({**r, "spec": spec})
            continue
        results.append({**r, "spec": spec})

    print_separator("Multi-domain • Multi-language — tổng quan từng tập")

    hdr = (
        f"{'Domain':<26} {'Lang':<5} {'Rows':>6} {'Skip':>5} "
        f"{'L0':>6} {'L1':>6} {'L2':>6} "
        f"{'µchr':>6} {'µwrd':>5}  Path"
    )
    print(hdr)
    print("-" * len(hdr))

    by_lang: Counter[str] = Counter()
    by_domain: Counter[str] = Counter()
    label_global: Counter[int] = Counter()
    total_rows_used = 0

    for r in results:
        if "error" in r:
            print(
                f"{'ERROR':<26} {'—':<5} {'—':>6} {'—':>5} "
                f"{'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>5}  "
                f"{r.get('path', '')}"
            )
            continue
        lab = r["labels"]
        n = r["rows_used"]
        total_rows_used += n
        by_lang[r["language"]] += n
        by_domain[r["domain"]] += n
        label_global.update(lab)

        l0, l1, l2 = lab.get(0, 0), lab.get(1, 0), lab.get(2, 0)
        rel = Path(r["path"]).relative_to(ROOT)

        print(
            f"{r['domain']:<26} {r['language']:<5} {r['rows_read']:>6} "
            f"{r['skipped']:>5} {l0:>6} {l1:>6} {l2:>6} "
            f"{r['mean_chars']:>6.0f} {r['mean_words']:>5.1f}  {rel}"
        )

        langs_report = r["langs_in_file"]
        if isinstance(langs_report, dict) and len(langs_report) > 1:
            print(f"    → language trong file: {langs_report}")

    if missing:
        print_separator("File không đọc được / không tồn tại")
        for p in missing:
            print(f"  - {p}")

    print_separator("Tổng hợp theo ngôn ngữ (rows dùng được)")

    for lang, cnt in sorted(by_lang.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {lang:>4}  {cnt:>6} rows  ({pct(cnt, total_rows_used)} tổng)")

    print_separator("Tổng hợp theo domain")

    for dom, cnt in sorted(by_domain.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {dom:<30} {cnt:>6} rows")

    print_separator("Phân bố nhãn toàn bộ corpus (rows dùng được)")

    for code, name in enumerate(LABEL_NAMES):
        c = label_global.get(code, 0)
        print(f"  {name:<18} {c:>6}  ({pct(c, total_rows_used)})")

    print(f"\n  Tổng số mẫu (sau skip): {total_rows_used}")
    combos = {(r["domain"], r["language"]) for r in results if "error" not in r}
    print(f"  Số (domain × language): {len(combos)}")

    out_csv = ROOT / "dataset_statistics_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "domain",
                "language",
                "rows_read",
                "skipped",
                "label_0",
                "label_1",
                "label_2",
                "mean_chars",
                "median_chars",
                "mean_words",
                "median_words",
                "path",
            ]
        )
        for r in results:
            if "error" in r:
                continue
            lab = r["labels"]
            w.writerow(
                [
                    r["domain"],
                    r["language"],
                    r["rows_read"],
                    r["skipped"],
                    lab.get(0, 0),
                    lab.get(1, 0),
                    lab.get(2, 0),
                    f"{r['mean_chars']:.2f}",
                    f"{r['median_chars']:.2f}",
                    f"{r['mean_words']:.2f}",
                    f"{r['median_words']:.2f}",
                    str(r["path"]),
                ]
            )

    print(f"\nĐã ghi chi tiết từng tập vào: {out_csv.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
