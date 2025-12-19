import csv
import argparse
from collections import defaultdict
from pathlib import Path

CLASSES = ["PALM", "FIST", "NONE"]
IDX = {c: i for i, c in enumerate(CLASSES)}

def norm_label(x):
    return str(x).strip().upper() if x is not None else ""

def load_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            yield row

def build_confusions(in_csv: str):
    zones = defaultdict(lambda: [[0,0,0],[0,0,0],[0,0,0]])  # 3x3 per zone
    for row in load_rows(Path(in_csv)):
        zone = str(row.get("zone", "")).strip()
        gt = norm_label(row.get("ground_truth", ""))
        pred = norm_label(row.get("gesture_pred", ""))
        if gt not in ("PALM", "FIST"):
            continue  # require labeled GT
        pred3 = pred if pred in ("PALM", "FIST") else "NONE"
        gi, pi = IDX[gt], IDX[pred3]
        zones[zone][gi][pi] += 1
    return zones

def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*prec*rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

def metrics_from_matrix(M):
    total = sum(sum(r) for r in M)
    correct = sum(M[i][i] for i in range(3))
    overall_acc = (correct / total) if total else 0.0

    per = {}
    for i, label in enumerate(CLASSES):
        tp = M[i][i]
        fp = sum(M[k][i] for k in range(3)) - tp
        fn = sum(M[i][k] for k in range(3)) - tp
        prec, rec, f1 = prf(tp, fp, fn)
        per[label] = {"TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "f1": f1}

    macro_f1 = (per["PALM"]["f1"] + per["FIST"]["f1"]) / 2.0
    return per, overall_acc, macro_f1, total

def add_mats(A, B):
    return [[A[i][j] + B[i][j] for j in range(3)] for i in range(3)]

def format_row(zone, M, include_none=False):
    per, acc, macro_f1, total = metrics_from_matrix(M)
    row = {
        "zone": zone,
        "samples": total,
        "overall_accuracy": round(acc, 4),
        "macro_f1_palmpfist": round(macro_f1, 4),
        "PALM_precision": round(per["PALM"]["precision"], 4),
        "PALM_recall": round(per["PALM"]["recall"], 4),
        "PALM_f1": round(per["PALM"]["f1"], 4),
        "FIST_precision": round(per["FIST"]["precision"], 4),
        "FIST_recall": round(per["FIST"]["recall"], 4),
        "FIST_f1": round(per["FIST"]["f1"], 4),
    }
    if include_none:
        row.update({
            "NONE_precision": round(per["NONE"]["precision"], 4),
            "NONE_recall": round(per["NONE"]["recall"], 4),
            "NONE_f1": round(per["NONE"]["f1"], 4),
        })
    return row

def sort_zone_key(z):  # numeric zones first
    z = str(z)
    return (0, int(z)) if z.isdigit() else (1, z)

def build_rows(in_csv, include_none=False):
    zones = build_confusions(in_csv)
    if not zones:
        return []

    rows = []
    total_M = [[0,0,0],[0,0,0],[0,0,0]]
    for z in sorted(zones.keys(), key=sort_zone_key):
        M = zones[z]
        rows.append(format_row(z, M, include_none=include_none))
        total_M = add_mats(total_M, M)

    rows.append(format_row("TOTAL", total_M, include_none=include_none))
    return rows

# -------- Pretty table (terminal) --------
def print_table(rows, include_none=False, ascii_only=False):
    # Column spec
    cols = [
        ("zone", "Zone"),
        ("samples", "Samples"),
        ("overall_accuracy", "OverallAcc"),
        ("macro_f1_palmpfist", "MacroF1(P+F)"),
        ("PALM_precision", "PALM_P"),
        ("PALM_recall", "PALM_R"),
        ("PALM_f1", "PALM_F1"),
        ("FIST_precision", "FIST_P"),
        ("FIST_recall", "FIST_R"),
        ("FIST_f1", "FIST_F1"),
    ]
    if include_none:
        cols += [("NONE_precision","NONE_P"),("NONE_recall","NONE_R"),("NONE_f1","NONE_F1")]

    # Compute widths
    def cell_str(k, v):
        return str(v)
    widths = {}
    for k, title in cols:
        widths[k] = max(len(title), max(len(cell_str(k, r.get(k,""))) for r in rows))

    # Borders
    if ascii_only:
        TL, TR, BL, BR = "+", "+", "+", "+"
        H, V, TSEP, MSEP, BSEP = "-", "|", "+", "+", "+"
    else:
        TL, TR, BL, BR = "┌", "┐", "└", "┘"
        H, V, TSEP, MSEP, BSEP = "─", "│", "┬", "┼", "┴"

    def horiz(left, mid, right):
        parts = []
        for i, (k, _) in enumerate(cols):
            parts.append(H * (widths[k] + 2))
        return left + mid.join(parts) + right

    def render_row(r):
        parts = []
        for k, _ in cols:
            parts.append(f" {str(r.get(k,'')):{widths[k]}} ")
        return V + V.join(parts) + V

    # Header + rows
    print(horiz(TL, TSEP, TR))
    header = {k: title for k, title in cols}
    print(render_row(header))
    print(horiz("├" if not ascii_only else "+", MSEP, "┤" if not ascii_only else "+"))
    for r in rows[:-1]:
        print(render_row(r))
    print(horiz("├" if not ascii_only else "+", MSEP, "┤" if not ascii_only else "+"))
    print(render_row(rows[-1]))  # TOTAL
    print(horiz(BL, BSEP, BR))

# -------- Writers --------
def write_csv(rows, out_path, include_none=False):
    fieldnames = [
        "zone","samples","overall_accuracy","macro_f1_palmpfist",
        "PALM_precision","PALM_recall","PALM_f1",
        "FIST_precision","FIST_recall","FIST_f1",
    ]
    if include_none:
        fieldnames += ["NONE_precision","NONE_recall","NONE_f1"]
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in fieldnames})

def write_pretty_delimited(rows, out_path, include_none=False, delim="|"):
    # human-friendly “CSV-like” with custom delimiter and a separator header line
    headers = [
        ("zone","Zone"),
        ("samples","Samples"),
        ("overall_accuracy","OverallAcc"),
        ("macro_f1_palmpfist","MacroF1(P+F)"),
        ("PALM_precision","PALM_P"),
        ("PALM_recall","PALM_R"),
        ("PALM_f1","PALM_F1"),
        ("FIST_precision","FIST_P"),
        ("FIST_recall","FIST_R"),
        ("FIST_f1","FIST_F1"),
    ]
    if include_none:
        headers += [("NONE_precision","NONE_P"),("NONE_recall","NONE_R"),("NONE_f1","NONE_F1")]

    # widths for alignment
    widths = {}
    for k, title in headers:
        widths[k] = max(len(title), max(len(str(r.get(k,""))) for r in rows))
    with Path(out_path).open("w", encoding="utf-8") as f:
        # Excel hint for custom delimiter
        f.write(f"sep={delim}\n")
        # header
        f.write(delim.join(title.ljust(widths[k]) for k, title in headers) + "\n")
        # underline
        f.write(delim.join("-"*widths[k] for k, _ in headers) + "\n")
        # rows
        for r in rows:
            f.write(delim.join(str(r.get(k,"")).ljust(widths[k]) for k, _ in headers) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Print per-zone standard metrics (3-class: PALM, FIST, NONE)")
    ap.add_argument("--file", default="metrics_samples.csv", help="Input CSV (metrics_samples.csv)")
    ap.add_argument("--out", default="", help="Output standard CSV path (optional)")
    ap.add_argument("--pretty-out", default="", help="Output ‘pretty’ delimited table (e.g., .txt) with separators")
    ap.add_argument("--pretty-delim", default="|", help="Delimiter for pretty output (default: |)")
    ap.add_argument("--include-none", action="store_true", help="Include NONE class metrics")
    ap.add_argument("--ascii", action="store_true", help="ASCII borders in terminal (fallback if Unicode not supported)")
    args = ap.parse_args()

    rows = build_rows(args.file, include_none=args.include_none)
    if not rows:
        print("No data found. Make sure metrics_samples.csv has zone, ground_truth, gesture_pred.")
        return

    print_table(rows, include_none=args.include_none, ascii_only=args.ascii)

    if args.out:
        write_csv(rows, args.out, include_none=args.include_none)
        print(f"Wrote CSV: {args.out}")
    if args.pretty_out:
        write_pretty_delimited(rows, args.pretty_out, include_none=args.include_none, delim=args.pretty_delim)
        print(f"Wrote pretty table: {args.pretty_out}")

if __name__ == "__main__":
    main()