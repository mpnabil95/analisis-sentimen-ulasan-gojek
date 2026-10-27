#!/usr/bin/env python3
"""Audit dataset sentimen Gojek secara read-only.

Usage:
    python -m src.audit_dataset_gojek --raw data/raw/ulasan_gojek_raw.csv
    python -m src.audit_dataset_gojek --raw data/raw/ulasan_gojek_raw.csv --labeled data/processed/ulasan_gojek_labeled.csv

Reads input data without changing it; writes a JSON audit report.
Requires pandas and scikit-learn for the full audit; does not retrain models.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def cleaning_text(text: str) -> str:
    """Mirror cleaningText plus casefoldingText in the repository notebook."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"RT[\s]", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[0-9]+", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def overlap_report(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    a = set(left["text_stem"].astype(str))
    b = set(right["text_stem"].astype(str))
    common = a & b
    # Counts include repeated examples present on both sides.
    return {
        "overlapping_unique_texts": len(common),
        "left_rows_with_overlap": int(left["text_stem"].isin(common).sum()),
        "right_rows_with_overlap": int(right["text_stem"].isin(common).sum()),
    }


def split_audit(d: pd.DataFrame, test_size: float, seed: int = 42) -> dict:
    left, right = train_test_split(
        d, test_size=test_size, random_state=seed, stratify=d["polarity"]
    )
    return {
        "train_rows": len(left),
        "test_rows": len(right),
        "train_test_overlap": overlap_report(left, right),
    }


def run(raw_path: Path, labeled_path: Path | None) -> dict:
    raw = pd.read_csv(raw_path)
    expected = {"content", "score", "at", "thumbsUpCount"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"Kolom raw yang hilang: {sorted(expected - set(raw.columns))}")
    raw_valid = raw.dropna(subset=["content"])
    raw_valid = raw_valid[raw_valid["content"].astype(str).str.strip().ne("")]
    raw_unique = raw_valid.drop_duplicates(subset=["content"]).copy()
    raw_unique["clean_fold"] = raw_unique["content"].map(cleaning_text)
    clean_nonempty = raw_unique.loc[raw_unique["clean_fold"].ne("")]
    report = {
        "source_files": {"raw": str(raw_path), "labeled": str(labeled_path) if labeled_path else None},
        "raw": {
            "rows": len(raw),
            "valid_content_rows": len(raw_valid),
            "unique_exact_content": len(raw_unique),
            "extra_exact_duplicate_rows": len(raw_valid) - len(raw_unique),
            "unique_cleaned_casefolded_nonempty": int(clean_nonempty["clean_fold"].nunique()),
            "extra_distinct_contents_colliding_after_cleaning_casefolding": int(
                clean_nonempty.duplicated(subset=["clean_fold"]).sum()
            ),
            "distinct_raw_contents_empty_after_cleaning": int(raw_unique["clean_fold"].eq("").sum()),
            "star_counts": {str(k): int(v) for k, v in raw["score"].value_counts().sort_index().items()},
        },
    }
    if labeled_path:
        labeled = pd.read_csv(labeled_path)
        required = {"text_stem", "polarity"}
        if not required.issubset(labeled.columns):
            raise ValueError(f"Kolom labeled yang hilang: {sorted(required - set(labeled.columns))}")
        labeled = labeled.dropna(subset=["text_stem", "polarity"]).copy()
        labeled["text_stem"] = labeled["text_stem"].astype(str)
        orig = len(labeled)
        dedup = labeled.drop_duplicates(subset=["text_stem"], keep="first")
        label_conflicts = int((labeled.groupby("text_stem")["polarity"].nunique() > 1).sum())
        report["labeled"] = {
            "rows": orig,
            "unique_text_stem": len(dedup),
            "extra_duplicate_text_stem_rows": orig - len(dedup),
            "unique_classes": sorted(labeled["polarity"].unique().tolist()),
            "class_counts_before": {str(k): int(v) for k, v in labeled["polarity"].value_counts().items()},
            "class_counts_after_dedup": {str(k): int(v) for k, v in dedup["polarity"].value_counts().items()},
            "duplicate_groups_with_conflicting_labels": label_conflicts,
            "at_least_10000_after_dedup": len(dedup) >= 10000,
            "original_splits": {
                "scheme_1_80_20": split_audit(labeled, 0.20),
                "scheme_2_70_30": split_audit(labeled, 0.30),
            },
            "deduplicated_splits": {
                "scheme_1_80_20": split_audit(dedup, 0.20),
                "scheme_2_70_30": split_audit(dedup, 0.30),
            },
        }
        # Scheme 3 has a further 10% validation split of its training partition.
        dl_label_map = {"positif": 0, "netral": 1, "negatif": 2}
        if not set(labeled["polarity"]).issubset(dl_label_map):
            raise ValueError("Label di luar kelas positif, netral, negatif.")
        base_train, base_test = train_test_split(
            labeled, test_size=0.20, random_state=42,
            stratify=labeled["polarity"].map(dl_label_map),
        )
        inner_train, validation = train_test_split(
            base_train, test_size=0.10, random_state=42,
            stratify=base_train["polarity"].map(dl_label_map),
        )
        report["labeled"]["original_splits"]["scheme_3_80_20"] = {
            "train_rows_before_validation_split": len(base_train),
            "test_rows": len(base_test),
            "inner_train_rows": len(inner_train),
            "validation_rows": len(validation),
            "train_validation_overlap": overlap_report(inner_train, validation),
            "train_test_overlap_using_inner_train": overlap_report(inner_train, base_test),
            "validation_test_overlap": overlap_report(validation, base_test),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/ulasan_gojek_raw.csv")
    parser.add_argument("--labeled", type=Path, default=Path(__file__).resolve().parents[1] / "data/processed/ulasan_gojek_labeled.csv", help="Dataset berlabel hasil notebook")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results/metrics/audit_dataset_gojek_report_new.json")
    args = parser.parse_args()
    report = run(args.raw, args.labeled)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nLaporan audit tersimpan di: {args.output}")
    if args.labeled and not report["labeled"]["at_least_10000_after_dedup"]:
        print("PERINGATAN: Sampel setelah deduplikasi kurang dari 10.000!")


if __name__ == "__main__":
    main()
