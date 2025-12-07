import os
import time
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import json
import pygame

from AI.src.face_rec_cam import FaceRecognitionCam
from AI.src.face_add_cam import (
    RAW_FOLDER,
    PROCESSED_FOLDER,
    MODEL_PATH,
    OUTPUT_CLASSIFIER,
    IMAGES_PER_ORIENTATION,
    ORIENTATION_ORDER,
    NUM_IMAGES,
    CAPTURE_INTERVAL,
    DETECT_SCALE,
    MARGIN,
    MediaPipeFaceDetector,
    expand_and_clip,
    _run_align,
)

from AI.src.face_orientation import FaceOrientation
from AI.src.face_center_check import check_face_in_ellipse
from AI.src.add_vietnamese_text import AddVietnameseText
from AI.src.classifier import Classifier
from AI.src.facenet import delete_classifier_model
from AI.src.speech import Speech

app = Flask(__name__)

# Đảm bảo thư mục raw tồn tại
os.makedirs(RAW_FOLDER, exist_ok=True)

# Đối tượng nhận diện khuôn mặt dùng chung cho API /api/recognize
face_rec_cam = FaceRecognitionCam()
face_rec_cam.check_model_loaded()

# Bộ nhớ các phiên thêm khuôn mặt (id -> session)
add_sessions = {}


def decode_base64_image(data_url: str):
    """
    data_url dạng 'data:image/jpeg;base64,...' -> frame BGR (numpy array)
    """
    if not data_url:
        return None
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    try:
        img_bytes = base64.b64decode(encoded)
    except Exception:
        return None
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame


def encode_image_to_base64(frame):
    """
    frame BGR (numpy) -> 'data:image/jpeg;base64,...'
    """
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode("utf-8")
    return "data:image/jpeg;base64," + b64


class FaceAddWebSession:
    """
    Chuyển logic trong face_add_cam.main() thành session xử lý từng frame
    được gửi từ trình duyệt.
    """
    def __init__(self, person_id: str):
        self.person_id = str(person_id).strip()
        self.person_folder = os.path.join(RAW_FOLDER, self.person_id)
        os.makedirs(self.person_folder, exist_ok=True)

        self.detector = MediaPipeFaceDetector(min_conf=0.5, long_range=False)
        self.face_orientation = FaceOrientation()
        self.speech = Speech()

        self.front_count = 0
        self.left_count = 0
        self.right_count = 0
        self.total_count = 0

        self.last_capture_time = time.time()
        self.last_speech_time = 0.0
        self.SPEECH_INTERVAL = 5.0
        self.last_target_ori = None

        self.done = False
        self.trained = False

        # 🔊 Vừa tạo session: yêu cầu đặt mặt trong khung
        try:
            self.speech.Trong_khung_start()
        except Exception:
            pass

    def get_target_orientation(self):
        """
        Quy định thứ tự: Front -> Left -> Right giống ORIENTATION_ORDER
        """
        if self.front_count < IMAGES_PER_ORIENTATION:
            return "Front"
        if self.left_count < IMAGES_PER_ORIENTATION:
            return "Left"
        if self.right_count < IMAGES_PER_ORIENTATION:
            return "Right"
        return None

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Xử lý 1 frame: detect -> kiểm tra elip + hướng -> quyết định chụp
        Trả về (frame_đã_vẽ, info_dict)
        """
        now = time.time()
        target_ori = self.get_target_orientation()
        if target_ori is None:
            # Đã chụp đủ ảnh
            self.done = True
            msg_done = f"Đã chụp đủ {NUM_IMAGES} ảnh (10 thẳng, 10 trái, 10 phải)."
            frame_bgr = AddVietnameseText.add_vietnamese_text(
                frame_bgr, msg_done, (10, 30),
                font_size=36, font_color=(0, 255, 0)
            )
            return frame_bgr, {
                "target_orientation": None,
                "front_count": self.front_count,
                "left_count": self.left_count,
                "right_count": self.right_count,
                "total_count": self.total_count,
                "message": msg_done,
                "done": True,
            }

        # 🔊 Nếu pha chụp thay đổi, phát âm thanh
        if target_ori != self.last_target_ori:
            try:
                if not pygame.mixer.get_busy():
                    if target_ori == "Front":
                        self.speech.Nhin_thang_start()
                    elif target_ori == "Left":
                        self.speech.Xoay_trai_start()
                    elif target_ori == "Right":
                        self.speech.Xoay_phai_start()
                    self.last_speech_time = now
            except Exception:
                pass
            self.last_target_ori = target_ori

        # Chuẩn bị ảnh để detect
        detect_in = frame_bgr
        if DETECT_SCALE != 1.0:
            detect_in = cv2.resize(frame_bgr, None, fx=DETECT_SCALE, fy=DETECT_SCALE)

        save_frame = frame_bgr.copy()
        faces = self.detector.detect_faces(detect_in)
        num_face = len(faces)
        message = ""

        if num_face == 0:
            frame_bgr = AddVietnameseText.add_vietnamese_text(
                frame_bgr, "Không phát hiện khuôn mặt", (10, 30),
                font_size=32, font_color=(0, 0, 255)
            )
            message = "Không phát hiện khuôn mặt"
            if (now - self.last_speech_time >= self.SPEECH_INTERVAL):
                try:
                    if not pygame.mixer.get_busy():
                        self.speech.Trong_khung_start()
                        self.last_speech_time = now
                except Exception:
                    pass

        elif num_face > 1:
            frame_bgr = AddVietnameseText.add_vietnamese_text(
                frame_bgr, "Có nhiều hơn 1 khuôn mặt trong khung hình", (10, 30),
                font_size=32, font_color=(0, 0, 255)
            )
            message = "Có nhiều hơn 1 khuôn mặt"
            if (now - self.last_speech_time >= self.SPEECH_INTERVAL):
                try:
                    if not pygame.mixer.get_busy():
                        self.speech.Trong_khung_start()
                        self.last_speech_time = now
                except Exception:
                    pass

        else:
            # Có đúng 1 khuôn mặt
            dx, dy, dw, dh = faces[0]["box"]
            sx = frame_bgr.shape[1] / detect_in.shape[1]
            sy = frame_bgr.shape[0] / detect_in.shape[0]
            x = int(round(dx * sx))
            y = int(round(dy * sy))
            w = int(round(dw * sx))
            h = int(round(dh * sy))

            # Nới rộng + clip bbox
            x, y, w, h = expand_and_clip(
                x, y, w, h,
                frame_bgr.shape[1], frame_bgr.shape[0],
                margin=MARGIN
            )

            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            status_text, status_color, inner_rect, outer_rect, cen, axes = \
                check_face_in_ellipse(frame_bgr, (x, y, w, h))

            # Vẽ elip hướng dẫn
            try:
                cv2.ellipse(frame_bgr, cen, axes, 0, 0, 360, (220, 220, 220), 5)
            except Exception:
                pass

            # Tính hướng mặt (Front / Left / Right / ...)
            face_ori_label = self.face_orientation.face_orientation_detection(faces)
            cv2.putText(
                frame_bgr, face_ori_label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
            )

            in_frame_ok = (status_text == "OK")
            orientation_ok = (
                (target_ori == "Front" and face_ori_label == "Front") or
                (target_ori == "Left" and face_ori_label == "Left") or
                (target_ori == "Right" and face_ori_label == "Right")
            )

            if in_frame_ok and orientation_ok and not self.done and (now - self.last_capture_time >= CAPTURE_INTERVAL):
                if target_ori == "Front" and self.front_count < IMAGES_PER_ORIENTATION:
                    self.front_count += 1
                elif target_ori == "Left" and self.left_count < IMAGES_PER_ORIENTATION:
                    self.left_count += 1
                elif target_ori == "Right" and self.right_count < IMAGES_PER_ORIENTATION:
                    self.right_count += 1

                self.total_count += 1
                image_path = os.path.join(
                    self.person_folder,
                    f"{self.person_id}_{self.total_count:03}.png"
                )
                cv2.imwrite(image_path, save_frame)
                self.last_capture_time = now

                message = (
                    f"Đã chụp {self.total_count}/{NUM_IMAGES} ảnh "
                    f"(Thẳng: {self.front_count}/{IMAGES_PER_ORIENTATION}, "
                    f"Trái: {self.left_count}/{IMAGES_PER_ORIENTATION}, "
                    f"Phải: {self.right_count}/{IMAGES_PER_ORIENTATION})"
                )
            else:
                message = f"Hãy nhìn {target_ori} và đặt mặt trong vùng elip."

            # Dòng text tổng hợp
            frame_bgr = AddVietnameseText.add_vietnamese_text(
                frame_bgr,
                f"Target: {target_ori} | Front {self.front_count}/10  Left {self.left_count}/10  Right {self.right_count}/10",
                (10, 70),
                font_size=30,
                font_color=(0, 255, 0) if in_frame_ok and orientation_ok else (0, 255, 255)
            )

        return frame_bgr, {
            "target_orientation": target_ori,
            "front_count": self.front_count,
            "left_count": self.left_count,
            "right_count": self.right_count,
            "total_count": self.total_count,
            "message": message,
            "done": self.done,
        }


# ============ ROUTES GIAO DIỆN ============

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recognize")
def recognize_page():
    return render_template("recognize.html")


@app.route("/add-face")
def add_face_page():
    return render_template("add_face.html")


# ============ API: NHẬN DIỆN REALTIME ============

@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    data = request.get_json(silent=True) or {}
    img_data = data.get("image")

    frame = decode_base64_image(img_data)
    if frame is None:
        return jsonify({"ok": False, "error": "Không giải mã được ảnh"}), 400

    processed_frame, current_id = face_rec_cam.process_frame(frame)
    out_img = encode_image_to_base64(processed_frame)
    stable_id = face_rec_cam.get_most_common_id()

    return jsonify({
        "ok": True,
        "current_id": str(current_id),
        "stable_id": str(stable_id),
        "image": out_img,
    })


# ============ API: THÊM KHUÔN MẶT REALTIME ============

@app.route("/api/add_face_stream", methods=["POST"])
def api_add_face_stream():
    data = request.get_json(silent=True) or {}
    person_id = (data.get("person_id") or "").strip()
    img_data = data.get("image")

    if not person_id:
        return jsonify({"ok": False, "error": "Thiếu person_id"}), 400
    if not img_data:
        return jsonify({"ok": False, "error": "Thiếu image"}), 400

    frame = decode_base64_image(img_data)
    if frame is None:
        return jsonify({"ok": False, "error": "Không giải mã được ảnh"}), 400

    session = add_sessions.get(person_id)
    if session is None:
        session = FaceAddWebSession(person_id)
        add_sessions[person_id] = session

    frame_out, info = session.process_frame(frame)
    out_img = encode_image_to_base64(frame_out)

    # Khi đã chụp đủ ảnh và chưa train -> align + train giống face_add_cam.main()
    if info["done"] and not session.trained:
        try:
            _run_align(person_id)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Lỗi align dữ liệu: {e}"}), 500

        try:
            delete_classifier_model()
        except Exception:
            pass

        try:
            Classifier(PROCESSED_FOLDER, MODEL_PATH, OUTPUT_CLASSIFIER)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Lỗi huấn luyện classifier: {e}"}), 500

        # Reload mô hình nhận diện để dùng classifier mới
        global face_rec_cam
        face_rec_cam = FaceRecognitionCam()
        face_rec_cam.check_model_loaded()

        session.trained = True
        info["message"] = info["message"] + " | Đã align & huấn luyện lại classifier."

    return jsonify({
        "ok": True,
        "image": out_img,
        **info
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
