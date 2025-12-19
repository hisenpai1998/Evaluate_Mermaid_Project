import time
import csv
import cv2
import sys
from pathlib import Path
from dataclasses import dataclass, field

from measure_metrics import GestureMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_REC_DIR = REPO_ROOT / "hand-recognition"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HAND_REC_DIR) not in sys.path:
    sys.path.insert(0, str(HAND_REC_DIR))
import hand_recognition as hr

ZONE_DURATION_SEC = 60  
SAMPLE_PERIOD_SEC = 0.5  
TOTAL_ZONES = 6        
# pred == FIST → case 3 if GT=FIST, else case 1
# pred == PALM → case 2 if GT=PALM, else case 4
# pred is None/other:
# GT == PALM → case 5 (miss PALM)
# GT == FIST → case 6 (miss FIST)
def classify_case(pred: str | None, gt: str | None) -> int | None:
    if pred == "FIST":
        return 3 if gt == "FIST" else 1
    if pred == "PALM":
        return 2 if gt == "PALM" else 4
    # pred None or other
    if gt == "PALM":
        return 5
    if gt == "FIST":
        return 6
    return None

@dataclass
class ZoneMetrics:
    zone: int
    gm: GestureMetrics = field(default_factory=lambda: GestureMetrics(ground_truth=None, window=30))
    rows: list = field(default_factory=list)
    samples: int = 0
    cases_count: dict = field(default_factory=lambda: {i: 0 for i in range(1, 7)})

    def add_detection(self, record: dict, pred: str | None, gt: str | None):
        self.rows.append(record)
        self.samples += 1
        cid = classify_case(pred, gt)
        if cid is not None:
            self.cases_count[cid] += 1

    def summary_dict(self):
        s = self.gm.summary()
        # Accuracy per-zone computed from 6 cases:
        correct = self.cases_count[2] + self.cases_count[3]
        total = sum(self.cases_count.values())
        acc_pct = (correct / total * 100.0) if total else 0.0
        return {
            "zone": self.zone,
            "samples": self.samples,
            "accuracy_pct": acc_pct,  # per-zone accuracy
            "consistency_pct": s["consistency_pct"],
            "camera_to_interaction_ms": s["camera_to_interaction_ms"],
            "gesture_processing_ms": s["gesture_processing_ms"],
            "camera_to_detect_ms": s["camera_to_detect_ms"],
            "case_1": self.cases_count[1],
            "case_2": self.cases_count[2],
            "case_3": self.cases_count[3],
            "case_4": self.cases_count[4],
            "case_5": self.cases_count[5],
            "case_6": self.cases_count[6],
        }

def run_sampling(out_csv: str = "./metrics_samples.csv", zones_csv: str = "./zone_summaries.csv"):
    det = hr.HandRecognizer()
    det.run()

    current_gt = None
    running = False
    zone_idx = 1
    rows_global = []
    zones: dict[int, ZoneMetrics] = {i: ZoneMetrics(zone=i) for i in range(1, TOTAL_ZONES + 1)}

    zone_elapsed = 0.0
    last_tick = None
    last_sample_ts = None 

    try:
        while zone_idx <= TOTAL_ZONES:
            z = zones[zone_idx]
            z.gm.ground_truth = (lambda _t, ref=current_gt: ref)

            if running:
                if last_tick is None:
                    last_tick = time.time()
                else:
                    now = time.time()
                    zone_elapsed += (now - last_tick)
                    last_tick = now
            else:
                last_tick = None

            zone_remaining = max(0, int(ZONE_DURATION_SEC - zone_elapsed))
            if zone_elapsed >= ZONE_DURATION_SEC:
                running = False
                last_tick = None
                zone_elapsed = 0.0
                last_sample_ts = None
                zone_idx += 1
                if zone_idx > TOTAL_ZONES:
                    break
                continue

            frame = det.get_frame()
            pred = det.get_gesture()
            pos = det.get_position()

            # UI overlay
            if frame is not None:
                cv2.putText(frame, f"ZONE: {zone_idx}/{TOTAL_ZONES}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"GT(f/p): {current_gt or 'NONE'}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Pred: {pred or 'None'}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"RUN(s/x): {'ON' if running else 'OFF'}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(frame, f"Remain: {zone_remaining:02d}s", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                cv2.putText(frame, f"Samples (zone): {z.samples}", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
                cv2.putText(frame, "Keys: f=FIST, p=PALM, s=Start, x=Stop(pause), q=Quit", (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                cv2.imshow("Experiment (6 zones, 60s each, 10Hz sampling)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('f'):
                    current_gt = "FIST"
                elif k == ord('p'):
                    current_gt = "PALM"
                elif k == ord('s'):
                    running = True
                elif k == ord('x'):
                    running = False
                elif k == ord('q'):
                    break

            now_ts = time.time()
            sample_due = (
                running and
                current_gt in ("FIST", "PALM") and
                (last_sample_ts is None or (now_ts - last_sample_ts) >= SAMPLE_PERIOD_SEC)
            )

            if sample_due:
                last_sample_ts = now_ts

                if pred in ("FIST", "PALM"):
                    z.gm.on_detect(pred)
                    z.gm.on_alert()

                case_id = classify_case(pred, current_gt)
                rec = {
                    "zone": zone_idx,
                    "timestamp": now_ts,
                    "gesture_pred": pred if pred else "",
                    "ground_truth": current_gt,
                    "case_id": case_id if case_id is not None else "",
                    "pos_x": pos[0] if pos else "",
                    "pos_y": pos[1] if pos else "",

                }
                rows_global.append(rec)
                z.add_detection(rec, pred, current_gt)

            time.sleep(0.005)
    finally:
        det.stop()
        cv2.destroyAllWindows()
        # Write per-detection CSV (global)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            if rows_global:
                w = csv.DictWriter(f, fieldnames=list(rows_global[0].keys()))
                w.writeheader()
                w.writerows(rows_global)
            else:

                f.write("zone,timestamp,gesture_pred,ground_truth,case_id,pos_x,pos_y\n")

        # Write per-zone summaries (includes per-zone accuracy from case counts)
        zone_summaries = [zones[i].summary_dict() for i in range(1, TOTAL_ZONES + 1)]
        with open(zones_csv, "w", newline="", encoding="utf-8") as fz:
            if zone_summaries:
                w = csv.DictWriter(fz, fieldnames=list(zone_summaries[0].keys()))
                w.writeheader()
                w.writerows(zone_summaries)
            else:
                fz.write("zone,samples,accuracy_pct,consistency_pct,camera_to_interaction_ms,gesture_processing_ms,camera_to_detect_ms,case_1,case_2,case_3,case_4,case_5,case_6\n")

        total_samples = sum(z.samples for z in zones.values())
        print(f"Total samples collected: {total_samples}")
        print(f"Saved detection CSV: {out_csv}")
        print(f"Saved per-zone summaries: {zones_csv}")

if __name__ == "__main__":
    run_sampling(out_csv="./metrics_samples.csv", zones_csv="./zone_summaries.csv")
