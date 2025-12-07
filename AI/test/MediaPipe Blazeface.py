import cv2
import time
import mediapipe as mp

# --------- Tham số tuỳ chọn ----------
CAMERA_INDEX = 0             # đổi sang 1 nếu dùng camera USB
FRAME_WIDTH, FRAME_HEIGHT = 720, 1080
MODEL_LONG_RANGE = False      # False: gần (0.5–2m), True: xa (đến ~5m)
MIN_CONF = 0.5                # ngưỡng tin cậy
# -------------------------------------

def draw_detections(frame, detections):
    h, w = frame.shape[:2]

    for det in detections:
        # Vẽ bbox
        rbox = det.location_data.relative_bounding_box
        x1 = int(max(0, rbox.xmin) * w)
        y1 = int(max(0, rbox.ymin) * h)
        x2 = int(min(1.0, rbox.xmin + rbox.width) * w)
        y2 = int(min(1.0, rbox.ymin + rbox.height) * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Chỉ vẽ 3 landmark: 2 mắt + mũi (thường là index 0,1,2)
        rkp = det.location_data.relative_keypoints
        keep_idx = (0, 1, 2)  # 0: mắt phải, 1: mắt trái, 2: mũi (thứ tự MediaPipe)
        for i in keep_idx:
            if i < len(rkp):
                cx = int(rkp[i].x * w)
                cy = int(rkp[i].y * h)
                # gợi ý: tô màu khác nhau cho dễ nhìn (mắt = xanh, mũi = đỏ)
                color = (0, 255, 0) if i in (0, 1) else (0, 0, 255)
                cv2.circle(frame, (cx, cy), 3, color, -1)

        # Hiển thị điểm tin cậy
        if det.score:
            cv2.putText(frame, f"{det.score[0]:.2f}",
                        (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)


def main():
    mp_fd = mp.solutions.face_detection
    model_sel = 1 if MODEL_LONG_RANGE else 0

    # Khởi tạo camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # giảm trễ

    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera index {CAMERA_INDEX}")

    # Khởi tạo FaceDetection
    with mp_fd.FaceDetection(model_selection=model_sel,
                             min_detection_confidence=MIN_CONF) as face_det:
        prev = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # BGR -> RGB (MediaPipe yêu cầu)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_det.process(rgb)

            if res.detections:
                draw_detections(frame, res.detections)

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev) if now > prev else 0.0
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("MediaPipe BlazeFace - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
