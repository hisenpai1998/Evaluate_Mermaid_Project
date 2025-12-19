import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV = "metrics_samples.csv"  # expects columns: zone, gesture_pred, ground_truth

CLASSES = ["PALM", "FIST", "NONE"]
IDX = {c: i for i, c in enumerate(CLASSES)}

def norm_label(x):
    if x is None:
        return ""
    return str(x).strip().upper()

def load_data(path=CSV):
    df = pd.read_csv(path)
    # normalize text
    df["zone"] = df["zone"].astype(str)
    df["ground_truth"] = df["ground_truth"].map(norm_label)
    df["gesture_pred"] = df["gesture_pred"].map(norm_label)
    # map predictions to 3 classes (NONE if not PALM/FIST)
    df["pred3"] = df["gesture_pred"].where(df["gesture_pred"].isin(["PALM", "FIST"]), "NONE")
    # ground truth is only PALM/FIST in your experiment; drop unlabeled rows if any
    df = df[df["ground_truth"].isin(["PALM", "FIST"])]
    return df

def confusion_3x3(df_zone):
    M = np.zeros((3, 3), dtype=int)
    for _, r in df_zone.iterrows():
        gt = r["ground_truth"]
        pr = r["pred3"]
        gi = IDX[gt]
        pi = IDX[pr]
        M[gi, pi] += 1
    return M

def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * prec * rec) / (prec + rec) if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else np.nan
    return prec, rec, f1

def metrics_from_matrix(M):
    # Per-class metrics
    per_class = {}
    total = M.sum()
    acc = np.trace(M) / total if total else np.nan
    for i, label in enumerate(CLASSES):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        tn = total - tp - fp - fn
        prec, rec, f1 = prf(tp, fp, fn)
        per_class[label] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision": prec, "recall": rec, "f1": f1,
        }
    # Macro‑F1 (PALM & FIST only; NONE usually has no GT -> NaN)
    macro_f1 = np.nanmean([per_class["PALM"]["f1"], per_class["FIST"]["f1"]])
    return per_class, acc, macro_f1

def plot_confusion(M, title, out_path):
    plt.figure(figsize=(4.2, 3.8))
    ax = sns.heatmap(M, annot=True, fmt="d", cmap="Blues",
                     xticklabels=CLASSES, yticklabels=CLASSES,
                     cbar=False, annot_kws={"fontsize": 10})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_confusion_grid(zone_to_M, out_path):
    zones = list(zone_to_M.keys())
    rows, cols = 2, 3
    plt.figure(figsize=(cols*4.2, rows*3.8))
    for idx, z in enumerate(zones):
        ax = plt.subplot(rows, cols, idx+1)
        M = zone_to_M[z]
        sns.heatmap(M, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    cbar=False, annot_kws={"fontsize": 8}, ax=ax)
        ax.set_title(f"Zone {z}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_bars(per_class, title, out_path):
    # per_class: dict {"PALM": {...}, "FIST": {...}, "NONE": {...}}
    labels = CLASSES
    prec = [per_class[c]["precision"] for c in labels]
    rec  = [per_class[c]["recall"] for c in labels]
    f1   = [per_class[c]["f1"] for c in labels]

    def _nan_to_zero(a): return [0.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in a]

    prec = _nan_to_zero(prec)
    rec  = _nan_to_zero(rec)
    f1   = _nan_to_zero(f1)

    x = np.arange(len(labels))
    w = 0.25

    plt.figure(figsize=(6.8, 4.0))
    plt.bar(x - w, prec, width=w, label="Precision")
    plt.bar(x,      rec,  width=w, label="Recall")
    plt.bar(x + w,  f1,   width=w, label="F1")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    if not os.path.exists(CSV):
        print(f"File not found: {CSV}")
        return

    df = load_data(CSV)
    zones = sorted(df["zone"].unique(), key=lambda z: int(z) if str(z).isdigit() else z)

    zone_rows = []
    zone_to_M = {}
    # Per-zone processing
    for z in zones:
        dz = df[df["zone"] == z]
        M = confusion_3x3(dz)
        zone_to_M[z] = M
        per_class, acc, macro_f1 = metrics_from_matrix(M)

        # Save per-zone bars
        plot_bars(per_class, f"Zone {z} — Precision/Recall/F1", f"zone_metrics_bars_zone{z}.png")
        # Collect metrics row
        row = {
            "zone": z,
            "samples": int(M.sum()),
            "overall_accuracy": round(acc if not np.isnan(acc) else 0.0, 4),
            "macro_f1_palmpfist": round(macro_f1 if not np.isnan(macro_f1) else 0.0, 4),
        }
        for cls in CLASSES:
            r = per_class[cls]
            row[f"{cls}_TP"] = r["TP"]
            row[f"{cls}_FP"] = r["FP"]
            row[f"{cls}_FN"] = r["FN"]
            row[f"{cls}_precision"] = round(r["precision"], 4) if not np.isnan(r["precision"]) else ""
            row[f"{cls}_recall"] = round(r["recall"], 4) if not np.isnan(r["recall"]) else ""
            row[f"{cls}_f1"] = round(r["f1"], 4) if not np.isnan(r["f1"]) else ""
        zone_rows.append(row)

    # Grid of confusion heatmaps for all zones
    if zones:
        plot_confusion_grid(zone_to_M, "confusion_grid.png")

    # TOTAL (overall)
    if zones:
        Mtot = sum(zone_to_M[z] for z in zones)
        per_class_T, acc_T, macro_f1_T = metrics_from_matrix(Mtot)
        plot_confusion(Mtot, "Overall Confusion (3-class)", "confusion_total.png")

        total_row = {
            "zone": "TOTAL",
            "samples": int(Mtot.sum()),
            "overall_accuracy": round(acc_T if not np.isnan(acc_T) else 0.0, 4),
            "macro_f1_palmpfist": round(macro_f1_T if not np.isnan(macro_f1_T) else 0.0, 4),
        }
        for cls in CLASSES:
            r = per_class_T[cls]
            total_row[f"{cls}_TP"] = r["TP"]
            total_row[f"{cls}_FP"] = r["FP"]
            total_row[f"{cls}_FN"] = r["FN"]
            total_row[f"{cls}_precision"] = round(r["precision"], 4) if not np.isnan(r["precision"]) else ""
            total_row[f"{cls}_recall"] = round(r["recall"], 4) if not np.isnan(r["recall"]) else ""
            total_row[f"{cls}_f1"] = round(r["f1"], 4) if not np.isnan(r["f1"]) else ""
        zone_rows.append(total_row)

    # Write metrics table
    if zone_rows:
        cols = list(zone_rows[0].keys())
        pd.DataFrame(zone_rows, columns=cols).to_csv("zone_three_class_metrics.csv", index=False)

    print("Done. Generated:")
    print(" - confusion_grid.png")
    print(" - confusion_total.png")
    print(" - zone_metrics_bars_zone{1..N}.png")
    print(" - zone_three_class_metrics.csv")

if __name__ == "__main__":
    main()