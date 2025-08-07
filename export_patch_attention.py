import argparse
import glob
import os
from typing import List, Dict

import h5py
import pandas as pd


def _gather_slide_rows(raw_dir: str, slide_id: str, label, pred_label: str, h5_pattern: str) -> List[Dict]:
    """Collect per-patch attention info for a single slide.

    Parameters
    ----------
    raw_dir : str
        Root directory containing raw heatmap results grouped by label and slide id.
    slide_id : str
        Identifier of the slide.
    label : str or int
        Ground-truth label for the slide. Used to locate the subdirectory.
    pred_label : str
        Predicted label for the slide.
    h5_pattern : str
        Glob pattern for locating the h5 file with attention scores.

    Returns
    -------
    list of dict
        Rows containing slide_id, labels, coordinates and attention score.
    """
    slide_dir = os.path.join(raw_dir, str(label), slide_id)
    h5_files = glob.glob(os.path.join(slide_dir, h5_pattern))
    if not h5_files:
        raise FileNotFoundError(f"No h5 file matching {h5_pattern} in {slide_dir}")

    rows: List[Dict] = []
    h5_path = h5_files[0]
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        scores = f["attention_scores"][:]

    for (x, y), score in zip(coords, scores):
        rows.append(
            {
                "slide_id": slide_id,
                "label": label,
                "pred": pred_label,
                "x": int(x),
                "y": int(y),
                "attention": float(score),
            }
        )
    return rows


def main(results_csv: str, raw_dir: str, output_csv: str, h5_pattern: str) -> None:
    """Generate a CSV of patch-level attention scores.

    Parameters
    ----------
    results_csv : str
        Path to CSV produced by create_heatmaps.py containing slide-level predictions.
    raw_dir : str
        Root directory containing raw heatmap h5 files.
    output_csv : str
        Destination CSV path.
    h5_pattern : str
        Glob pattern for locating h5 files within each slide directory.
    """
    slide_df = pd.read_csv(results_csv)
    all_rows: List[Dict] = []
    for _, row in slide_df.iterrows():
        slide_id = row["slide_id"]
        label = row.get("label", "")
        pred_label = row.get("Pred_0", "")
        try:
            slide_rows = _gather_slide_rows(raw_dir, slide_id, label, pred_label, h5_pattern)
            all_rows.extend(slide_rows)
        except FileNotFoundError as err:
            print(f"Warning: {err}")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export patch-level attention scores to CSV after heatmap generation.")
    parser.add_argument(
        "--results_csv",
        required=True,
        help="CSV from heatmaps/results containing slide-level predictions",
    )
    parser.add_argument(
        "--raw_dir",
        required=True,
        help="Directory with raw heatmap results (e.g. heatmaps/heatmap_raw_results/EXP_CODE)",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to save the generated CSV",
    )
    parser.add_argument(
        "--h5_pattern",
        default="*_blockmap.h5",
        help="Glob pattern for heatmap h5 files within each slide directory",
    )
    args = parser.parse_args()
    main(args.results_csv, args.raw_dir, args.output_csv, args.h5_pattern)
