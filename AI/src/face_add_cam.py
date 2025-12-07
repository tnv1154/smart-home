import os
import cv2
import time
import shutil
import numpy as np
import mediapipe as mp  # Dùng MediaPipe Face Detection
import pygame           # 🔊 THÊM: để kiểm tra mixer đang bận hay không

# ====== Import các mô-đun nội bộ của bạn ======
from AI.src.add_vietnamese_text import AddVietnameseText
from AI.src.classifier import Classifier
from AI.src.facenet import delete_classifier_model
from AI.src.face_center_check import check_face_in_ellipse
from AI.src.face_orientation import FaceOrientation
from AI.src.speech import Speech  # dùng phát âm thanh hướng dẫn
# Cố gắng tương thích cả hai kiểu đặt tên hàm align
try:
    from AI.src.align_data_mediapipe import ailgn_data as _align_runner
except Exception:
    try:
        from AI.src.align_data_mediapipe import align_data as _align_runner
    except Exception:
        _align_runner = None
# =============================================

# ====== Cấu hình dự án của bạn ======
Base_path = "E:/PythonProjectMain/AI"
print(f"Thư mục gốc : {Base_path}")

RAW_FOLDER = os.path.join(Base_path, "DataSet", "FaceData", "raw")
PROCESSED_FOLDER = os.path.join(Base_path, "DataSet", "FaceData", "processed")
MODEL_PATH = os.path.join(Base_path, "Models", "20180402-114759.pb")
OUTPUT_CLASSIFIER = os.path.join(Base_path, "Models", "classifier.pkl")

# Thông số camera / hiển thị / chụp
CAMERA_INDEX = 0
DISPLAY_SCALE = 0.7     # chỉ áp dụng cho khung HIỂN THỊ
DETECT_SCALE = 0.5      # downscale đều (fx=fy) để detect cho nhanh (0.5 là hợp lý)
MARGIN = 0.05           # nới rộng bbox để ôm trọn khuôn mặt

# Cấu hình số ảnh theo hướng mặt
IMAGES_PER_ORIENTATION = 10
ORIENTATION_ORDER = ("Front", "Left", "Right")
NUM_IMAGES = IMAGES_PER_ORIENTATION * len(ORIENTATION_ORDER)  # = 30
CAPTURE_INTERVAL = 0.25  # giây giữa 2 lần chụp liên tiếp
# ====================================


# ====== Detector MediaPipe, bổ sung keypoints để dùng cho FaceOrientation ======
class MediaPipeFaceDetector:
    def __init__(self, min_conf=0.5, long_range=False):
        # model_selection=0 cho cự ly gần; 1 cho cự ly xa
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1 if long_range else 0,
            min_detection_confidence=min_conf
        )

    def detect_faces(self, image_bgr):
        """
        Trả về danh sách dict:
          {
            'box': [x, y, w, h],
            'keypoints': {'left_eye':(x,y), 'right_eye':(x,y), 'nose':(x,y)},
            'score': float
          }
        Tính trên ảnh đầu vào image_bgr.
        """
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        out = []

        if not res.detections:
            return out

        for d in res.detections:
            r = d.location_data.relative_bounding_box
            # clip 0..1 rồi đổi ra pixel
            xmin = max(0.0, r.xmin)
            ymin = max(0.0, r.ymin)
            xmax = min(1.0, r.xmin + r.width)
            ymax = min(1.0, r.ymin + r.height)
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # keypoints: 0=right_eye, 1=left_eye, 2=nose_tip
            kps = d.location_data.relative_keypoints
            kp_dict = {
                'left_eye': (int(kps[1].x * w), int(kps[1].y * h)),
                'right_eye': (int(kps[0].x * w), int(kps[0].y * h)),
                'nose': (int(kps[2].x * w), int(kps[2].y * h)),
            }

            score = float(d.score[0]) if d.score else 0.0
            out.append({
                'box': [x1, y1, bw, bh],
                'keypoints': kp_dict,
                'score': score
            })
        return out

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
# ================================================================


def expand_and_clip(x, y, w, h, W, H, margin=MARGIN):
    """
    Nới rộng bbox theo cạnh lớn + clip vào khung hình.
    Trả về (x, y, w, h) mới (int).
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    s = max(w, h) * (1.0 + margin * 2.0)
    x1 = int(round(cx - s / 2.0))
    y1 = int(round(cy - s / 2.0))
    x2 = int(round(cx + s / 2.0))
    y2 = int(round(cy + s / 2.0))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


class FaceAdd:
    def __init__(self, id_employee):
        self.face_add = main(id_employee)


def _run_align(person_id: str):
    if _align_runner is None:
        print("[CẢNH BÁO] Không tìm thấy hàm align trong AI.src.align_data_mediapipe. Bỏ qua bước cắt khuôn mặt.")
        return
    try:
        _align_runner(person_id)
    except Exception as e:
        print(f"[LỖI] Gọi align thất bại: {e}")


def main(id_employee: str):
    """Thêm mới nhân viên: chụp ảnh → align → trích xuất đặc trưng → train classifier"""
    person_id = str(id_employee).strip()
    if not person_id:
        print("ID nhân viên trống!")
        return

    person_folder = os.path.join(RAW_FOLDER, person_id)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder, exist_ok=True)
    else:
        print(f"Thư mục cho id {person_id} đã tồn tại: {person_folder}")
        overwrite = input("Tiếp tục và ghi đè dữ liệu (y/n): ").strip().lower()
        if overwrite != "y":
            print("Hủy thao tác")
            return

    print(f"Chuẩn bị chụp {NUM_IMAGES} ảnh cho id {person_id}...")
    print("Yêu cầu: 10 ảnh nhìn thẳng, 10 ảnh xoay trái, 10 ảnh xoay phải.")
    print("Bắt đầu chụp...  (Nhấn q để thoát)")

    detector = MediaPipeFaceDetector(min_conf=0.5, long_range=False)
    face_orientation = FaceOrientation()
    speech = Speech()

    # Khởi tạo camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    time.sleep(1.5)

    # 🔊 Vừa mở camera xong: yêu cầu đặt khuôn mặt trong khung (chỉ 1 lần)
    speech.Trong_khung_start()

    # Đếm riêng theo từng hướng mặt
    front_count = 0
    left_count = 0
    right_count = 0
    count = 0  # tổng số ảnh đã chụp (1..30)

    last_capture_time = time.time()
    last_speech_time = 0.0
    SPEECH_INTERVAL = 5.0  # giây giữa 2 lần nói
    last_target_ori = None  # để phát âm thanh mỗi khi ĐỔI HƯỚNG

    def get_target_orientation():
        """Trả về hướng mặt đang cần chụp tiếp theo."""
        nonlocal front_count, left_count, right_count
        if front_count < IMAGES_PER_ORIENTATION:
            return "Front"
        if left_count < IMAGES_PER_ORIENTATION:
            return "Left"
        if right_count < IMAGES_PER_ORIENTATION:
            return "Right"
        return None

    while True:
        target_ori = get_target_orientation()
        if target_ori is None:
            print("Đã chụp đủ 30 ảnh (10 thẳng, 10 trái, 10 phải).")
            break

        # 🔊 Nếu pha cần chụp thay đổi (Front → Left → Right), phát âm thanh tương ứng
        if target_ori != last_target_ori:
            # Chỉ phát nếu mixer không đang bận (tránh chồng tiếng)
            if not pygame.mixer.get_busy():
                if target_ori == "Front":
                    speech.Nhin_thang_start()
                elif target_ori == "Left":
                    speech.Xoay_trai_start()
                elif target_ori == "Right":
                    speech.Xoay_phai_start()
                last_speech_time = time.time()
            last_target_ori = target_ori

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Không thể đọc frame từ camera")
            continue

        # Ảnh để detect (downscale đều, giữ tỉ lệ)
        detect_in = frame
        if DETECT_SCALE != 1.0:
            detect_in = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)

        save_frame = frame.copy()  # lưu ảnh gốc vào RAW
        faces = detector.detect_faces(detect_in)
        num_face = len(faces)
        now = time.time()

        # ===== Trường hợp không có hoặc nhiều hơn 1 khuôn mặt =====
        if num_face == 0:
            frame = AddVietnameseText.add_vietnamese_text(
                frame, "Không phát hiện khuôn mặt", (10, 30),
                font_size=50, font_color=(255, 0, 0)
            )
            # nhắc lại đặt mặt vào khung sau mỗi SPEECH_INTERVAL
            if (now - last_speech_time >= SPEECH_INTERVAL) and (not pygame.mixer.get_busy()):
                speech.Trong_khung_start()
                last_speech_time = now

        elif num_face > 1:
            frame = AddVietnameseText.add_vietnamese_text(
                frame, "Có nhiều hơn 1 khuôn mặt trong khung hình", (10, 30),
                font_size=50, font_color=(255, 0, 0)
            )
            if (now - last_speech_time >= SPEECH_INTERVAL) and (not pygame.mixer.get_busy()):
                speech.Trong_khung_start()
                last_speech_time = now

        # ===== Trường hợp có đúng 1 khuôn mặt =====
        else:
            # bbox trên detect_in → scale ngược về frame gốc
            dx, dy, dw, dh = faces[0]['box']
            sx = frame.shape[1] / detect_in.shape[1]
            sy = frame.shape[0] / detect_in.shape[0]
            x = int(round(dx * sx))
            y = int(round(dy * sy))
            w = int(round(dw * sx))
            h = int(round(dh * sy))

            # nới bbox + clip để ôm trọn khuôn mặt
            x, y, w, h = expand_and_clip(
                x, y, w, h,
                frame.shape[1], frame.shape[0],
                margin=MARGIN
            )

            # vẽ bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Kiểm tra vị trí khuôn mặt trong elip (logic trong face_center_check)
            status_text, status_color, inner_rect, outer_rect, cen, axes = check_face_in_ellipse(
                frame, (x, y, w, h)
            )

            # CHỈ VẼ ELIP (không vẽ 2 hình chữ nhật)
            cv2.ellipse(frame, cen, axes, 0, 0, 360, (220, 220, 220), 5)

            # Tính hướng mặt (Front / Left / Right / ...)
            face_ori_label = face_orientation.face_orientation_detection(faces)
            cv2.putText(
                frame, face_ori_label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
            )

            # Điều kiện vị trí mặt trong vùng OK (theo elip)
            in_frame_ok = (status_text == "OK")

            # Điều kiện hướng mặt đúng với pha đang chụp
            orientation_ok = (
                (target_ori == "Front" and face_ori_label == "Front") or
                (target_ori == "Left" and face_ori_label == "Left") or
                (target_ori == "Right" and face_ori_label == "Right")
            )

            # ==== LOGIC CHỤP ẢNH ====
            if in_frame_ok and orientation_ok and now - last_capture_time >= CAPTURE_INTERVAL:
                # Chỉ chụp khi ở đúng vùng OK và đúng hướng
                if target_ori == "Front" and front_count < IMAGES_PER_ORIENTATION:
                    front_count += 1
                elif target_ori == "Left" and left_count < IMAGES_PER_ORIENTATION:
                    left_count += 1
                elif target_ori == "Right" and right_count < IMAGES_PER_ORIENTATION:
                    right_count += 1

                # Dùng biến count làm STT 1..30 để đặt tên file id_0xx.png
                count += 1
                image_path = os.path.join(
                    person_folder,
                    f"{person_id}_{count:03}.png"  # ví dụ: 2_001.png, 2_015.png
                )
                cv2.imwrite(image_path, save_frame)
                last_capture_time = now
                print(
                    f"Đã chụp {count}/{NUM_IMAGES} ảnh "
                    f"(Front: {front_count}/{IMAGES_PER_ORIENTATION}, "
                    f"Left: {left_count}/{IMAGES_PER_ORIENTATION}, "
                    f"Right: {right_count}/{IMAGES_PER_ORIENTATION})"
                )
            else:
                # Nếu chưa trong vùng OK / sai hướng → có thể nhắc, nhưng phải CHẶN SPAM
                if (now - last_speech_time >= SPEECH_INTERVAL) and (not pygame.mixer.get_busy()):
                    if not in_frame_ok:
                        # chưa đúng vị trí trong khung elip
                        speech.Trong_khung_start()
                    else:
                        # trong khung nhưng sai hướng so với pha hiện tại
                        if target_ori == "Front" and face_ori_label != "Front":
                            speech.Nhin_thang_start()
                        elif target_ori == "Left" and face_ori_label != "Left":
                            speech.Xoay_trai_start()
                        elif target_ori == "Right" and face_ori_label != "Right":
                            speech.Xoay_phai_start()
                    last_speech_time = now

            # Hiển thị text yêu cầu + tiến độ
            viet_target = {
                "Front": "Nhìn thẳng",
                "Left": "Xoay trái",
                "Right": "Xoay phải"
            }[target_ori]

            frame = AddVietnameseText.add_vietnamese_text(
                frame,
                f"Yêu cầu: {viet_target}",
                (10, 30),
                font_size=50,
                font_color=(0, 255, 0)
            )
            frame = AddVietnameseText.add_vietnamese_text(
                frame,
                f"Đã chụp - Thẳng: {front_count}/10  Trái: {left_count}/10  Phải: {right_count}/10",
                (10, 90),
                font_size=40,
                font_color=(0, 255, 0)
            )

        # Hiển thị (chỉ resize khi DISPLAY, không ảnh hưởng detect)
        disp_w = int(frame.shape[1] * DISPLAY_SCALE)
        disp_h = int(frame.shape[0] * DISPLAY_SCALE)
        disp = cv2.resize(frame, (disp_w, disp_h))
        cv2.imshow("Face add", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("Chụp ảnh hoàn tất")

    # Tiến hành cắt khuôn mặt (align)
    print("Cắt khuôn mặt (align)...")
    _run_align(person_id)
    print("Cắt khuôn mặt hoàn tất")

    # Trích xuất đặc trưng & huấn luyện bộ phân loại
    print("Huấn luyện bộ phân loại...")
    Classifier(PROCESSED_FOLDER, MODEL_PATH, OUTPUT_CLASSIFIER)
    print("Thêm mới nhân viên hoàn tất")


def face_re_train(id):
    """
    Huấn luyện lại bộ phân loại khi nhân viên bị xóa:
    - Xóa thư mục processed/<id> (nếu có)
    - Xóa model classifier cũ
    - Train lại
    """
    employees_folder = os.path.join(PROCESSED_FOLDER, str(id))
    if os.path.exists(employees_folder):
        shutil.rmtree(employees_folder)
        print(f"Đã xóa thư mục: {employees_folder}")
    else:
        print(f"Thư mục không tồn tại: {employees_folder}")

    delete_classifier_model()
    Classifier(PROCESSED_FOLDER, MODEL_PATH, OUTPUT_CLASSIFIER)
    print("Đã huấn luyện lại bộ phân loại")


if __name__ == "__main__":
    emp_id = input("Nhập id nhân viên: ").strip()
    if emp_id:
        FaceAdd(emp_id)
    else:
        print("ID trống, thoát.")
