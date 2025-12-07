"""Cắt khuôn mặt"""

import os
import numpy as np
from PIL import Image
import mediapipe as mp  # dùng MediaPipe thay cho MTCNN

face_path = "E:/PythonProjectMain/AI/DataSet/FaceData"

# ----- Detector MediaPipe với API giống MTCNN.detect_faces -----
class MediaPipeFace:
    """
    Trả về danh sách dict giống MTCNN:
      [{'box': [x, y, w, h], 'confidence': float}]
    (Keypoints không cần cho script này)
    """
    def __init__(self, min_conf=0.5, long_range=False):
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=1 if long_range else 0,
            min_detection_confidence=min_conf
        )

    def detect_faces(self, image_rgb: np.ndarray):
        # image_rgb: HxWx3, RGB uint8 (đúng với np.asarray(PIL.Image.convert("RGB")))
        h, w = image_rgb.shape[:2]
        res = self.det.process(image_rgb)
        out = []
        if not res.detections:
            return out
        for d in res.detections:
            r = d.location_data.relative_bounding_box
            # clip 0..1 rồi quy đổi pixel
            xmin = max(0.0, r.xmin); ymin = max(0.0, r.ymin)
            xmax = min(1.0, r.xmin + r.width)
            ymax = min(1.0, r.ymin + r.height)
            x1 = int(xmin * w); y1 = int(ymin * h)
            x2 = int(xmax * w); y2 = int(ymax * h)
            bw = max(1, x2 - x1); bh = max(1, y2 - y1)
            conf = float(d.score[0]) if d.score else 0.0
            out.append({'box': [x1, y1, bw, bh], 'confidence': conf})
        return out

    def __del__(self):
        try:
            self.det.close()
        except Exception:
            pass
# ---------------------------------------------------------------

detector = MediaPipeFace()  # giữ biến tên 'detector' như cũ

# Số pixel muốn mở rộng xung quanh khuôn mặt
MARGIN = 30

class ailgn_data:
    def __init__(self, id):
        crop_faces_for_id(id)

def crop_faces_for_id(id_name: str):
    raw_dir = os.path.join(face_path, "raw", id_name)
    processed_dir = os.path.join(face_path, "processed", id_name)
    # Tạo thư mục đích nếu chưa có
    os.makedirs(processed_dir, exist_ok=True)
    for i in range(1, 31):  # ảnh từ 001 đến 030
        file_name = f"{id_name}_{i:03}"
        input_path = os.path.join(raw_dir, f"{file_name}.png")
        output_path = os.path.join(processed_dir, f"{file_name}.png")
        if not os.path.exists(input_path):
            print(f"[WARN] Không tìm thấy ảnh: {input_path}")
            continue
        # Mở ảnh
        image = Image.open(input_path).convert("RGB")
        image_array = np.asarray(image)  # RGB uint8
        # Phát hiện khuôn mặt
        results = detector.detect_faces(image_array)
        if len(results) == 0:
            print(f"[WARN] Không phát hiện khuôn mặt: {file_name}")
            continue
        # Lấy bounding box của khuôn mặt đầu tiên
        x, y, width, height = results[0]['box']
        x_new = max(0, x - MARGIN)
        y_new = max(0, y - MARGIN)
        x2_new = min(image.width,  x + width  + MARGIN)
        y2_new = min(image.height, y + height + MARGIN)
        # Crop ảnh với bounding box có margin và giới hạn trong ảnh
        if x_new >= x2_new or y_new >= y2_new:
            print(f"[WARN] BBOX không hợp lệ sau clip: {file_name}")
            continue
        face = image.crop((x_new, y_new, x2_new, y2_new))
        # Resize về 160x160 (như code gốc)
        face = face.resize((160, 160))
        # Lưu ảnh
        face.save(output_path, format="PNG")
        print(f"[INFO] Đã xử lý: {file_name}.png  MARGIN : {MARGIN}")

def main():
    id = input("Nhap id: ")
    crop_faces_for_id(id)
    print("Done!")

if __name__ == "__main__":
    main()
