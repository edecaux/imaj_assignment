#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from colors_utils import quantize_image_wu_rmse
import pandas as pd
import numpy as np
import os
import argparse
import sys

def format_palette(best_palette, max_colors=25):
    if best_palette is None:
        return [""] * max_colors
    pal = np.array(best_palette, copy=False)
    out = []
    if getattr(pal, "ndim", 0) == 2 and np.issubdtype(pal.dtype, np.number):
        for row in pal[:max_colors]:
            r, g, b = int(row[0]), int(row[1]), int(row[2])
            pct = float(row[3]) if row.size >= 4 else None
            hexc = f"#{r:02x}{g:02x}{b:02x}"
            out.append(f"{hexc}|{pct:.2f}" if pct is not None else hexc)
    elif getattr(pal, "ndim", 0) == 1 and all(isinstance(x, str) for x in pal):
        out = pal[:max_colors].tolist()
    else:
        for elem in list(pal)[:max_colors]:
            if isinstance(elem, str):
                out.append(elem)
            else:
                try:
                    r, g, b = int(elem[0]), int(elem[1]), int(elem[2])
                    pct = float(elem[3]) if len(elem) >= 4 else None
                    hexc = f"#{r:02x}{g:02x}{b:02x}"
                    out.append(f"{hexc}|{pct:.2f}" if pct is not None else hexc)
                except Exception:
                    out.append("")
    if len(out) < max_colors:
        out.extend([""] * (max_colors - len(out)))
    return out

def _process_image(path_str, max_colors, RMSE_THRESHOLD):
    # path_str may be "" for missing files
    if not path_str:
        return [""] * max_colors
    try:
        best_img, best_palette, best_k, k_list, rmse_list = quantize_image_wu_rmse(
            path_str, RMSE_THRESHOLD=RMSE_THRESHOLD, max_colors=max_colors
        )
        return format_palette(best_palette, max_colors)
    except Exception:
        return [""] * max_colors

def add_palette_columns(csv_path: Path,
                        input_folder: Path,
                        out_csv: Path,
                        filename_col: str = "image",
                        max_colors: int = 25,
                        RMSE_THRESHOLD: float = 3.3,
                        max_workers: int = None):
    df = pd.read_csv(csv_path)
    if filename_col not in df.columns:
        filename_col = "image" if "image" in df.columns else df.columns[0]

    cols = [f"color {i+1}" for i in range(max_colors)]
    for c in cols:
        df[c] = ""

    # prepare tasks (index, display name, resolved path or "")
    tasks = []
    for idx, row in df.iterrows():
        val = row.get(filename_col, "")
        fname = "" if pd.isna(val) else str(val).strip()
        img_path = ""
        if fname:
            p = Path(fname)
            if not p.is_absolute():
                p = input_folder / p
            if not p.exists():
                alt = input_folder / Path(fname).name
                if alt.exists():
                    p = alt
            img_path = str(p) if p.exists() else ""
        tasks.append((idx, fname, img_path))

    n = len(tasks)
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 4))

    # submit jobs and collect as they complete to print progress
    results = {}
    submitted = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for idx, fname, pathstr in tasks:
            fut = ex.submit(_process_image, pathstr, max_colors, RMSE_THRESHOLD)
            submitted[fut] = (idx, fname, pathstr)
        done_count = 0
        for fut in as_completed(submitted):
            idx, fname, pathstr = submitted[fut]
            palette_list = fut.result()
            results[idx] = palette_list
            done_count += 1
            pct = done_count * 100.0 / n if n else 100.0
            print(f"[{done_count}/{n}] idx={idx} file='{fname}' path='{pathstr or 'MISSING'}'  ({pct:.1f}%)", flush=True)

    # assign results to DataFrame (preserve order)
    for idx, palette_list in results.items():
        for i, val in enumerate(palette_list[:max_colors]):
            if val:
                df.at[idx, cols[i]] = val

    df.to_csv(out_csv, index=False)
    print("Done. Wrote output to:", out_csv, flush=True)
    return out_csv

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/data/imaj/database/filtered_dataset_final.csv")
    p.add_argument("--input-folder", default="/data/imaj/database/filtered_dataset_resized")
    p.add_argument("--out-csv", default="/tmp/filtered_with_palette.csv")
    p.add_argument("--max-colors", type=int, default=25)
    p.add_argument("--rmse", type=float, default=3.3)
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    try:
        out = add_palette_columns(
            csv_path=Path(args.csv),
            input_folder=Path(args.input_folder),
            out_csv=Path(args.out_csv),
            max_colors=args.max_colors,
            RMSE_THRESHOLD=args.rmse,
            max_workers=args.workers
        )
        sys.exit(0)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(2)