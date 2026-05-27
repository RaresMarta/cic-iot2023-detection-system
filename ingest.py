"""Data ingestion: load CICIoT2023 CSVs, deduplicate, subsample, cache to Parquet."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def run_ingest(
    attack_folders: list[str],
    dataset_dir: Path,
    x_columns: list[str],
    y_column: str,
    max_rows: int,
    seed: int,
) -> tuple[list, int, int, int, dict]:
    """Scan every attack folder, deduplicate, subsample, and return collected frames.

    Returns:
        frames        – list of per-folder polars DataFrames
        total_raw     – raw row count across all folders
        total_unique  – after deduplication
        total_kept    – after sampling cap
        class_stats   – {folder_name: rows_kept}
    """
    frames: list[pl.DataFrame] = []
    total_raw = total_unique = total_kept = 0
    class_stats: dict[str, int] = {}

    for folder in attack_folders:
        folder_path = dataset_dir / folder
        csvs = sorted(folder_path.glob("*.csv"))
        if not csvs:
            continue

        per_file = [
            pl.scan_csv(str(f), infer_schema_length=10_000)
            .select(x_columns)
            .with_columns(pl.lit(f.name).alias("source_csv"))
            for f in csvs
        ]
        df = pl.concat(per_file).collect()
        n_raw = df.height

        df = df.unique(subset=x_columns, keep="first")
        n_dedup = df.height

        if max_rows and df.height > max_rows:
            df = df.sample(n=max_rows, seed=seed, shuffle=True)
        n_kept = df.height

        total_raw    += n_raw
        total_unique += n_dedup
        total_kept   += n_kept

        df = df.with_columns(pl.lit(folder).alias(y_column))
        frames.append(df)
        class_stats[folder] = n_kept
        print(f"  {folder}: {n_raw:,} raw → {n_dedup:,} unique → {n_kept:,} kept")

    return frames, total_raw, total_unique, total_kept, class_stats


def plot_waterfall(
    total_raw: int,
    total_unique: int,
    total_kept: int,
    max_rows_per_class: int,
    n_folders: int,
) -> plt.Figure:
    """Return a waterfall figure showing the data-reduction stages."""
    total_dupes  = total_raw - total_unique
    sampling_cut = total_unique - total_kept
    dupe_pct     = total_dupes  / total_raw    * 100
    sample_pct   = sampling_cut / total_unique * 100

    labels   = ["Raw", "− Duplicates", "Unique", "− Sampling cap", "Sampled"]
    values   = [total_raw, -total_dupes, total_unique, -sampling_cut, total_kept]
    is_total = [True, False, True, False, True]

    cum = 0
    bottoms, heights, colors = [], [], []
    for v, total in zip(values, is_total):
        if total:
            col = "#4c9be8" if v == total_raw else ("#3aa17e" if v == total_kept else "#7aa6c2")
            bottoms.append(0)
            heights.append(v)
            colors.append(col)
            cum = v
        else:
            new = cum + v
            bottoms.append(new)
            heights.append(-v)
            colors.append("#e07a5f")
            cum = new

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, heights, bottom=bottoms, color=colors, edgecolor="black", linewidth=0.5)

    tops, running = [], 0
    for v, total in zip(values, is_total):
        running = v if total else running + v
        tops.append(running)
    for i in range(len(tops) - 1):
        ax.plot([i + 0.4, i + 0.6], [tops[i], tops[i]], "k--", linewidth=0.7)

    annotations = [
        f"{total_raw:,}",
        f"−{total_dupes:,}\n({dupe_pct:.1f}% of raw)",
        f"{total_unique:,}",
        f"−{sampling_cut:,}\n({sample_pct:.1f}% of unique)",
        f"{total_kept:,}\ncap {max_rows_per_class:,}/class",
    ]
    for bar, bottom, height, text in zip(bars, bottoms, heights, annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bottom + height + total_raw * 0.015,
            text, ha="center", va="bottom", fontsize=9,
        )

    ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.set_ylabel("Rows")
    ax.set_ylim(0, total_raw * 1.15)
    ax.set_title(f"CIC-IoT-2023 ingest waterfall — {n_folders} folders")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
