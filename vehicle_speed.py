import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

VIDEO_PATH  = "input.mp4"        
OUTPUT_PATH = "output_speed.mp4"
MODEL_NAME  = "yolov8m.pt"      
CONF_THRES  = 0.3


VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


SOURCE = np.array([
    [285, 110],   
    [430, 110],   
    [555, 350],   
    [125, 350],   
], dtype=np.float32)

TARGET_WIDTH  = 7.3    
TARGET_HEIGHT = 60.0   

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH, 0],
    [TARGET_WIDTH, TARGET_HEIGHT],
    [0, TARGET_HEIGHT],
], dtype=np.float32)

M = cv2.getPerspectiveTransform(SOURCE, TARGET)  

def to_bev(point_xy):
    """Transform an image point (x, y) to bird's-eye-view meters."""
    pt = np.array([[point_xy]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])

SMOOTH_SECONDS = 0.7   
MIN_TRACK_LEN  = 3     
MIN_DT_SECONDS = 0.4   
MAX_STEP_M     = 4.0   
MAX_PLAUSIBLE_KPH = 200  

def speed_color(kph):
    if kph < 60:   return (0, 255, 0)    
    if kph < 100:  return (0, 255, 255)   
    return (0, 0, 255)                    


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

               
                cx = (x1 + x2) / 2.0
                bottom_y = y2

              
                inside = cv2.pointPolygonTest(
                    SOURCE.astype(np.float32), (float(cx), float(bottom_y)), False
                ) >= 0
                if not inside:
                   
                    speeds.pop(tid, None)
                    history[tid].clear()
                else:
                    bx, by = to_bev((cx, bottom_y))

                  
                    if history[tid]:
                        pbx, pby, pf = history[tid][-1]
                        step = np.hypot(bx - pbx, by - pby)
                        if step > MAX_STEP_M * max(1, frame_idx - pf):
                            history[tid].clear()
                            speeds.pop(tid, None)
                    history[tid].append((bx, by, frame_idx))

                    
                    h = history[tid]
                    if len(h) >= MIN_TRACK_LEN:
                        bx0, by0, f0 = h[0]
                        bx1, by1, f1 = h[-1]
                        dt = (f1 - f0) / fps
                       
                        if dt >= MIN_DT_SECONDS:
                            dist_m = np.hypot(bx1 - bx0, by1 - by0)
                            kph = (dist_m / dt) * 3.6
                           
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

        
        for i, (txt, c) in enumerate([("< 60 km/h", (0, 255, 0)),
                                      ("60-100 km/h", (0, 255, 255)),
                                      ("> 100 km/h", (0, 0, 255))]):
            y = height - 70 + i * 22
            cv2.rectangle(frame, (10, y - 12), (26, y + 2), c, -1)
            cv2.putText(frame, txt, (32, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)

        
        cv2.polylines(frame, [SOURCE.astype(int)], True, (255, 0, 255), 2)

        writer.write(frame)
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"Done. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
