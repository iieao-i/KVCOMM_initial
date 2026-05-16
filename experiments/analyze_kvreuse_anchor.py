import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_events(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Input CSV is empty: {csv_path}")
    return rows


def build_summary(
    rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Counter, int]:
    grouped: Dict[Tuple[str, str], Dict[str, float]] = {}
    skip_reason_counter: Counter = Counter()
    max_step = 0

    for row in rows:
        ph_id = row.get("ph_id", "")
        anchor_msg = row.get("anchor_msg", "")
        key = (ph_id, anchor_msg)
        if key not in grouped:
            grouped[key] = {
                "candidate_count": 0.0,
                "selected_count": 0.0,
                "last_selected_step": -1.0,
                "weight_sum": 0.0,
                "weight_n": 0.0,
            }

        bucket = grouped[key]
        is_candidate = _to_int(row.get("is_candidate", "0"))
        is_selected = _to_int(row.get("is_selected", "0"))
        step = _to_int(row.get("step", "0"))
        weight = _to_float(row.get("weight", ""))
        skip_reason = row.get("skip_reason", "none") or "none"

        bucket["candidate_count"] += is_candidate
        bucket["selected_count"] += is_selected
        if is_selected:
            bucket["last_selected_step"] = max(bucket["last_selected_step"], step)
            bucket["weight_sum"] += weight
            bucket["weight_n"] += 1

        if skip_reason != "none":
            skip_reason_counter[skip_reason] += 1
        max_step = max(max_step, step)

    summary_rows: List[Dict[str, str]] = []
    for (ph_id, anchor_msg), bucket in grouped.items():
        candidate_count = int(bucket["candidate_count"])
        selected_count = int(bucket["selected_count"])
        last_selected_step = int(bucket["last_selected_step"])
        selected_rate = (selected_count / candidate_count) if candidate_count > 0 else 0.0
        idle_steps = (max_step - last_selected_step) if last_selected_step >= 0 else max_step
        avg_weight = (bucket["weight_sum"] / bucket["weight_n"]) if bucket["weight_n"] > 0 else 0.0
        summary_rows.append(
            {
                "ph_id": ph_id,
                "anchor_msg": anchor_msg,
                "candidate_count": str(candidate_count),
                "selected_count": str(selected_count),
                "selected_rate": f"{selected_rate:.6f}",
                "last_selected_step": str(last_selected_step),
                "idle_steps": str(int(idle_steps)),
                "avg_weight": f"{avg_weight:.6f}",
            }
        )

    summary_rows.sort(key=lambda x: int(x["selected_count"]), reverse=True)
    return summary_rows, skip_reason_counter, max_step


def save_summary_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    fields = [
        "ph_id",
        "anchor_msg",
        "candidate_count",
        "selected_count",
        "selected_rate",
        "last_selected_step",
        "idle_steps",
        "avg_weight",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_topn_bar(summary_rows: List[Dict[str, str]], output_path: Path, top_n: int) -> None:
    top = summary_rows[:top_n]
    labels = [f"{r['ph_id']}#{i}" for i, r in enumerate(top)]
    values = [int(r["selected_count"]) for r in top]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("selected_count")
    plt.title(f"KVReuse Anchor Hotness Top-{top_n}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_lorenz(summary_rows: List[Dict[str, str]], output_path: Path) -> None:
    counts = [int(r["selected_count"]) for r in summary_rows]
    if not counts:
        counts = [0]
    counts = sorted(counts)
    total = sum(counts)

    x, y = [0.0], [0.0]
    running = 0
    n = len(counts)
    for i, c in enumerate(counts, start=1):
        running += c
        x.append(i / n)
        y.append((running / total) if total > 0 else 0.0)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="Observed")
    plt.plot([0, 1], [0, 1], "--", label="Uniform")
    plt.xlabel("Cumulative share of anchors")
    plt.ylabel("Cumulative share of selections")
    plt.title("KVReuse Anchor Concentration (Lorenz Curve)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_rolling_concentration(
    rows: List[Dict[str, str]],
    output_path: Path,
    rolling_window: int,
) -> None:
    selected_rows = [r for r in rows if _to_int(r.get("is_selected", "0")) == 1]
    if not selected_rows:
        plt.figure(figsize=(12, 5))
        plt.title("No selected anchors found")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
        return

    selected_rows.sort(key=lambda r: _to_int(r.get("step", "0")))
    step_anchor_counts: Dict[int, Counter] = defaultdict(Counter)
    for r in selected_rows:
        step = _to_int(r.get("step", "0"))
        anchor_key = f"{r.get('ph_id', '')}||{r.get('anchor_msg', '')}"
        step_anchor_counts[step][anchor_key] += 1

    steps = sorted(step_anchor_counts.keys())
    top1_series, top5_series = [], []
    for idx, step in enumerate(steps):
        left = max(0, idx - rolling_window + 1)
        window_steps = steps[left : idx + 1]
        merged = Counter()
        for s in window_steps:
            merged.update(step_anchor_counts[s])
        total = sum(merged.values())
        if total == 0:
            top1_series.append(0.0)
            top5_series.append(0.0)
            continue
        freqs = sorted(merged.values(), reverse=True)
        top1_series.append(freqs[0] / total)
        top5_series.append(sum(freqs[:5]) / total)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, top1_series, label="Top1 share")
    plt.plot(steps, top5_series, label="Top5 share")
    plt.ylim(0.0, 1.0)
    plt.xlabel("step")
    plt.ylabel("share in rolling window")
    plt.title(f"KVReuse Rolling Anchor Concentration (window={rolling_window})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_global_hot_anchor_trends(
    rows: List[Dict[str, str]],
    summary_rows: List[Dict[str, str]],
    output_path: Path,
    top_k: int,
) -> None:
    top_anchors = [
        f"{r.get('ph_id', '')}||{r.get('anchor_msg', '')}"
        for r in summary_rows
        if int(r["selected_count"]) > 0
    ][:top_k]

    if not top_anchors:
        plt.figure(figsize=(12, 5))
        plt.title("No selected anchors found")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
        return

    selected_rows = [r for r in rows if _to_int(r.get("is_selected", "0")) == 1]
    selected_rows.sort(key=lambda r: _to_int(r.get("step", "0")))

    steps = sorted({_to_int(r.get("step", "0")) for r in selected_rows})
    per_step_counts: Dict[str, Counter] = {anchor: Counter() for anchor in top_anchors}
    for r in selected_rows:
        anchor_key = f"{r.get('ph_id', '')}||{r.get('anchor_msg', '')}"
        if anchor_key in per_step_counts:
            per_step_counts[anchor_key][_to_int(r.get("step", "0"))] += 1

    plt.figure(figsize=(12, 5))
    for rank, anchor_key in enumerate(top_anchors, start=1):
        running = 0
        cumulative = []
        for step in steps:
            running += per_step_counts[anchor_key][step]
            cumulative.append(running)
        ph_id, _ = anchor_key.split("||", 1)
        plt.plot(steps, cumulative, label=f"Top{rank}: {ph_id}")

    plt.xlabel("step")
    plt.ylabel("cumulative selected_count")
    plt.title(f"KVReuse Global Hot Anchor Trends (top={len(top_anchors)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_global_hot_anchor_rolling_share(
    rows: List[Dict[str, str]],
    summary_rows: List[Dict[str, str]],
    output_path: Path,
    top_k: int,
    rolling_window: int,
) -> None:
    top_anchors = [
        f"{r.get('ph_id', '')}||{r.get('anchor_msg', '')}"
        for r in summary_rows
        if int(r["selected_count"]) > 0
    ][:top_k]

    selected_rows = [r for r in rows if _to_int(r.get("is_selected", "0")) == 1]
    if not selected_rows or not top_anchors:
        plt.figure(figsize=(12, 5))
        plt.title("No selected anchors found")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
        return

    selected_rows.sort(key=lambda r: _to_int(r.get("step", "0")))
    step_anchor_counts: Dict[int, Counter] = defaultdict(Counter)
    for r in selected_rows:
        step = _to_int(r.get("step", "0"))
        anchor_key = f"{r.get('ph_id', '')}||{r.get('anchor_msg', '')}"
        step_anchor_counts[step][anchor_key] += 1

    steps = sorted(step_anchor_counts.keys())
    topk_total_series: List[float] = []

    for idx, _step in enumerate(steps):
        left = max(0, idx - rolling_window + 1)
        window_steps = steps[left : idx + 1]
        merged = Counter()
        for s in window_steps:
            merged.update(step_anchor_counts[s])

        total = sum(merged.values())
        if total == 0:
            topk_total_series.append(0.0)
            continue

        topk_total = sum(merged[anchor] for anchor in top_anchors)
        topk_total_series.append(topk_total / total)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, topk_total_series, linewidth=2.0, label=f"Top{len(top_anchors)} total")

    plt.ylim(0.0, 1.0)
    plt.xlabel("step")
    plt.ylabel("share in rolling window")
    plt.title(
        f"KVReuse Global Hot Anchor Rolling Share "
        f"(top={len(top_anchors)}, window={rolling_window})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KVReuse anchor hot/cold behavior.")
    parser.add_argument(
        "--events-csv",
        type=str,
        required=True,
        help="Path to kvreuse_anchor_events.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/anchor_analysis",
        help="Directory to save CSV and figures.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N anchors shown in bar chart.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=200,
        help="Rolling window size (in step bins) for concentration chart.",
    )
    parser.add_argument(
        "--trend-top-k",
        type=int,
        default=5,
        help="Global Top-K selected anchors shown in cumulative trend chart.",
    )
    args = parser.parse_args()

    events_csv = Path(args.events_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_events(events_csv)
    summary_rows, skip_reasons, max_step = build_summary(rows)

    summary_csv = output_dir / "kvreuse_anchor_summary.csv"
    save_summary_csv(summary_rows, summary_csv)

    plot_topn_bar(summary_rows, output_dir / "kvreuse_topn_hotness.png", args.top_n)
    plot_lorenz(summary_rows, output_dir / "kvreuse_lorenz_curve.png")
    plot_rolling_concentration(rows, output_dir / "kvreuse_rolling_concentration.png", args.rolling_window)
    plot_global_hot_anchor_trends(
        rows,
        summary_rows,
        output_dir / "kvreuse_global_hot_anchor_trends.png",
        args.trend_top_k,
    )
    plot_global_hot_anchor_rolling_share(
        rows,
        summary_rows,
        output_dir / "kvreuse_global_hot_anchor_rolling_share.png",
        args.trend_top_k,
        args.rolling_window,
    )

    print(f"[done] summary csv: {summary_csv}")
    print(f"[done] figures: {output_dir / 'kvreuse_topn_hotness.png'}")
    print(f"[done] figures: {output_dir / 'kvreuse_lorenz_curve.png'}")
    print(f"[done] figures: {output_dir / 'kvreuse_rolling_concentration.png'}")
    print(f"[done] figures: {output_dir / 'kvreuse_global_hot_anchor_trends.png'}")
    print(f"[done] figures: {output_dir / 'kvreuse_global_hot_anchor_rolling_share.png'}")
    print(f"[info] max_step={max_step}, anchors={len(summary_rows)}")
    if skip_reasons:
        print(f"[info] skip_reasons={dict(skip_reasons)}")


if __name__ == "__main__":
    main()
