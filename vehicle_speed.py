# ============================================================
# Vehicle Speed Detection — YOLOv8 + ByteTrack + Homography
# Designed for Google Colab (GPU runtime recommended)
#
# HOW TO USE IN COLAB:
#   1) Runtime -> Change runtime type -> GPU (T4 is fine)
#   2) In a cell, run:  !pip install -q ultralytics supervision
#   3) Upload your video (left sidebar -> Files) and set VIDEO_PATH
#   4) Adjust SOURCE points + TARGET_WIDTH/HEIGHT for your road (see notes)
#   5) Run:  !python vehicle_speed_colab.py   (or paste this whole file in a cell)
#   6) Download the annotated output video (OUTPUT_PATH)
# ============================================================

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ----------------------- SETTINGS ---------------------------
VIDEO_PATH  = "input.mp4"        # <-- your uploaded video
OUTPUT_PATH = "output_speed.mp4"
MODEL_NAME  = "yolov8m.pt"       # n/s/m — bigger = more accurate, slower
CONF_THRES  = 0.3

# Vehicle classes in COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ------------------- ROAD CALIBRATION -----------------------
# SOURCE: 4 pixel points in the video forming a rectangle ON THE ROAD,
# ordered: top-left, top-right, bottom-right, bottom-left.
# Pick points whose real-world size you know, e.g. a stretch of road:
#   - lane width in Pakistan/most countries ~ 3.5 m
#   - dashed lane markings: usually 3 m line + 6 m gap (9 m per cycle)
# TARGET_WIDTH / TARGET_HEIGHT: the real size (in METERS) of that rectangle.
#
# The defaults below are placeholders — you MUST tune them for your video,
# otherwise speeds will be wrong (relative speeds will still look plausible).
SOURCE = np.array([
    [285, 110],   # top-left  (median barrier, far end)
    [430, 110],   # top-right (left white edge line, far end)
    [555, 350],   # bottom-right (edge line, near end)
    [125, 350],   # bottom-left  (median barrier, near end)
], dtype=np.float32)

TARGET_WIDTH  = 7.3    # meters: 2 lanes x ~3.65 m
TARGET_HEIGHT = 60.0   # meters: tune by counting dash cycles (~9 m each)

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH, 0],
    [TARGET_WIDTH, TARGET_HEIGHT],
    [0, TARGET_HEIGHT],
], dtype=np.float32)

M = cv2.getPerspectiveTransform(SOURCE, TARGET)  # image px -> meters

def to_bev(point_xy):
    """Transform an image point (x, y) to bird's-eye-view meters."""
    pt = np.array([[point_xy]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])

# -------------------- SPEED SETTINGS ------------------------
SMOOTH_SECONDS = 0.7   # window over which speed is averaged
MIN_TRACK_LEN  = 3     # frames before we show a speed
MIN_DT_SECONDS = 0.4   # minimum elapsed time before trusting a speed
MAX_STEP_M     = 4.0   # max plausible movement (meters) per frame; more = ID switch
MAX_PLAUSIBLE_KPH = 200  # anything above this is discarded as noise

def speed_color(kph):
    if kph < 60:   return (0, 255, 0)     # green
    if kph < 100:  return (0, 255, 255)   # yellow
    return (0, 0, 255)                    # red

# ------------------------ MAIN ------------------------------
def main():
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Could not open {VIDEO_PATH}"
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    window = max(2, int(SMOOTH_SECONDS * fps))
    # tid -> deque of (bev_x, bev_y, frame_idx)
    history = defaultdict(lambda: deque(maxlen=window))
    speeds  = {}  # tid -> last computed kph

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRES,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            verbose=False,
        )

        r = results[0]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids   = r.boxes.id.int().cpu().numpy()
            clss  = r.boxes.cls.int().cpu().numpy()

            for (x1, y1, x2, y2), tid, cls_id in zip(boxes, ids, clss):
                cls_name = VEHICLE_CLASSES.get(int(cls_id), "Vehicle")

                # Bottom-center of box = where tires touch the road plane
                cx = (x1 + x2) / 2.0
                bottom_y = y2

                # GUARD 1: only measure speed INSIDE the calibration zone.
                # Outside it the homography extrapolates and gives garbage,
                # especially near the horizon.
                inside = cv2.pointPolygonTest(
                    SOURCE.astype(np.float32), (float(cx), float(bottom_y)), False
                ) >= 0
                if not inside:
                    # Still draw the box, but with no speed
                    speeds.pop(tid, None)
                    history[tid].clear()
                else:
                    bx, by = to_bev((cx, bottom_y))

                    # GUARD 2: reject teleports (ID switches / detector
                    # glitches). > MAX_STEP_M meters in one frame = discard
                    # the old history and start fresh.
                    if history[tid]:
                        pbx, pby, pf = history[tid][-1]
                        step = np.hypot(bx - pbx, by - pby)
                        if step > MAX_STEP_M * max(1, frame_idx - pf):
                            history[tid].clear()
                            speeds.pop(tid, None)
                    history[tid].append((bx, by, frame_idx))

                    # Compute speed over the smoothing window
                    h = history[tid]
                    if len(h) >= MIN_TRACK_LEN:
                        bx0, by0, f0 = h[0]
                        bx1, by1, f1 = h[-1]
                        dt = (f1 - f0) / fps
                        # GUARD 3: require a real time window before trusting
                        # the estimate (tiny dt amplifies pixel noise).
                        if dt >= MIN_DT_SECONDS:
                            dist_m = np.hypot(bx1 - bx0, by1 - by0)
                            kph = (dist_m / dt) * 3.6
                            # GUARD 4: drop physically impossible values
                            if kph <= MAX_PLAUSIBLE_KPH:
                                speeds[tid] = kph

                kph = speeds.get(tid)
                color = speed_color(kph) if kph is not None else (200, 200, 200)
                label = (f"{cls_name} ({kph:.0f} km/h)"
                         if kph is not None else cls_name)

                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, color, 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (p1[0], p1[1] - th - 8),
                              (p1[0] + tw + 4, p1[1]), color, -1)
                cv2.putText(frame, label, (p1[0] + 2, p1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Legend
        for i, (txt, c) in enumerate([("< 60 km/h", (0, 255, 0)),
                                      ("60-100 km/h", (0, 255, 255)),
                                      ("> 100 km/h", (0, 0, 255))]):
            y = height - 70 + i * 22
            cv2.rectangle(frame, (10, y - 12), (26, y + 2), c, -1)
            cv2.putText(frame, txt, (32, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)

        # (Optional) draw the calibration zone so you can tune SOURCE
        cv2.polylines(frame, [SOURCE.astype(int)], True, (255, 0, 255), 2)

        writer.write(frame)
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Done. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
