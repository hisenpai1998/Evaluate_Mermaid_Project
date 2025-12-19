import time
from collections import deque

class GestureMetrics:
    def __init__(self, ground_truth=None, window=30):
        # ground_truth: optional callable that returns the expected gesture label at time t
        self.ground_truth = ground_truth
        # Accuracy counters
        self.correct = 0
        self.total = 0
        # Consistency: majority of last N detections stays the same
        self.window = window
        self.recent = deque(maxlen=window)
        self.consistent_hits = 0
        self.consistency_checks = 0
        # Latencies
        self.frame_ts = 0.0           # camera frame timestamp
        self.detect_ts = 0.0          # when classifier outputs a gesture
        self.alert_ts = 0.0           # when your app reacts/emits
        self.frame_to_detect_ms = []  # camera → detect
        self.detect_to_alert_ms = []  # detect → alert (reaction)
        # Processing time per frame (classifier runtime)
        self.proc_times_ms = []       # gesture processing time

        # Last seen for stabilization
        self.last_gesture = None

    def mark_frame(self):
        self.frame_ts = time.time()

    def begin_processing(self):
        self._proc_start = time.time()

    def end_processing(self):
        self.proc_times_ms.append((time.time() - self._proc_start) * 1000.0)

    def on_detect(self, gesture: str):
        self.detect_ts = time.time()
        if self.frame_ts:
            self.frame_to_detect_ms.append((self.detect_ts - self.frame_ts) * 1000.0)

        # Accuracy (if you have labels; else skip)
        if self.ground_truth:
            expected = self.ground_truth(time.time())
            if expected in ("FIST", "PALM"):
                self.total += 1
                if gesture == expected:
                    self.correct += 1

        # Consistency window check
        if gesture:
            self.recent.append(gesture)
            if len(self.recent) == self.window:
                # Reliability = fraction of same labels in the window
                majority = max(set(self.recent), key=self.recent.count)
                frac_same = self.recent.count(majority) / self.window
                self.consistency_checks += 1
                # Count as "consistent" if ≥ 90% same in window
                if frac_same >= 0.9:
                    self.consistent_hits += 1

        self.last_gesture = gesture

    def on_alert(self):
        # Call when your code emits an action for the gesture
        self.alert_ts = time.time()
        if self.detect_ts:
            self.detect_to_alert_ms.append((self.alert_ts - self.detect_ts) * 1000.0)

    def summary(self):
        acc = (self.correct / self.total) * 100.0 if self.total else 0.0
        consistency = (self.consistent_hits / self.consistency_checks) * 100.0 if self.consistency_checks else 0.0
        cam_to_interaction = (sum(self.detect_to_alert_ms) / len(self.detect_to_alert_ms)) if self.detect_to_alert_ms else 0.0
        processing = (sum(self.proc_times_ms) / len(self.proc_times_ms)) if self.proc_times_ms else 0.0
        frame_to_detect = (sum(self.frame_to_detect_ms) / len(self.frame_to_detect_ms)) if self.frame_to_detect_ms else 0.0
        return {
            "accuracy_pct": acc,
            "consistency_pct": consistency,
            "camera_to_interaction_ms": cam_to_interaction,
            "gesture_processing_ms": processing,
            "camera_to_detect_ms": frame_to_detect,
        }