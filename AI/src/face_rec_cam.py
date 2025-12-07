import collections
from collections import Counter
import tensorflow as tf
import imutils
import os

import pickle
import numpy as np
import cv2

import time
import mediapipe as mp
from AI.src import facenet
from AI.src.add_vietnamese_text import AddVietnameseText

tf.compat.v1.enable_eager_execution()
IMAGE_SIZE = 160
INPUT_IMAGE_SIZE = 182
BASE_PATH = "E:/PythonProjectMain/AI"
CLASSIFIER_PATH = os.path.join(BASE_PATH, 'Models', 'classifier.pkl')
FACENET_MODEL_PATH = os.path.join(BASE_PATH, 'Models', '20180402-114759.pb')
MIN_FACE_HEIGHT_RATIO = 0.25

# ====== Detector MediaPipe mô phỏng API MTCNN.detect_faces ======
class MediaPipeFaceDetector:
    """
    .detect_faces(image_rgb) -> list[{'box':[x,y,w,h], 'confidence':float}]
    - image_rgb: HxWx3, RGB uint8 (đúng với small_frame bạn đang tạo)
    """
    def __init__(self, min_conf=0.5, long_range=False):
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=1 if long_range else 0,
            min_detection_confidence=min_conf
        )

    def detect_faces(self, image_rgb: np.ndarray):
        h, w = image_rgb.shape[:2]
        res = self.det.process(image_rgb)
        out = []
        if not res.detections:
            return out
        for d in res.detections:
            r = d.location_data.relative_bounding_box
            # clip 0..1 rồi đổi về pixel
            xmin = max(0.0, r.xmin); ymin = max(0.0, r.ymin)
            xmax = min(1.0, r.xmin + r.width)
            ymax = min(1.0, r.ymin + r.height)
            x1 = int(xmin * w); y1 = int(ymin * h)
            x2 = int(xmax * w); y2 = int(ymax * h)
            bw = max(1, x2 - x1); bh = max(1, y2 - y1)
            conf = float(d.score[0]) if d.score else 0.0
            out.append({'box':[x1, y1, bw, bh], 'confidence': conf})
        return out

    def __del__(self):
        try:
            self.det.close()
        except Exception:
            pass
# ===============================================================

class FaceNetModel:
    def __init__(self):
        self.is_loaded = False
        self.model = None
        self.class_names = None
        self.graph = None
        self.sess = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.detector = None

    def start_model(self):
        """Tải mô hình """
        if self.is_loaded:
            return

        print("Tải mô hình FaceNet...")
        start_time = time.time()

        # Tải mô hình classifier
        with open(CLASSIFIER_PATH, 'rb') as file:
            self.model, self.class_names = pickle.load(file)
        print("tải mô hình phân loại thành công")

        # Load graph FaceNet (TF1) trong TF2 qua compat v1
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            with tf.io.gfile.GFile(FACENET_MODEL_PATH, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            self.images_placeholder = self.graph.get_tensor_by_name("input:0")
            self.embeddings = self.graph.get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.shape[1]
        self.sess = tf.compat.v1.Session(graph=self.graph)

        # ✅ Dùng MediaPipe thay cho MTCNN (API tương tự)
        self.detector = MediaPipeFaceDetector(min_conf=0.5, long_range=False)

        self.is_loaded = True
        end_time = time.time()
        print(f"Tải mô hình thành công trong : {end_time - start_time:.2f} giây")

class FaceRecognitionCam:
    def __init__(self):
        self.model = FaceNetModel()
        self.id_arr = []
        self.person_detected = Counter()
        self.last_face_location = None
        self.last_face_id = None
        self.face_ttl = 0

    def check_model_loaded(self):
        """kiểm tra mô hình đã đc tải chưa"""
        if not self.model.is_loaded:
            self.model.start_model()

    def process_frame(self, frame):
        """Xử lý từng frame"""
        self.check_model_loaded()
        detected_id = None
        if frame is None:
            return frame, None

        # Resize frame
        frame = imutils.resize(frame, width=640)

        # Convert BGR -> RGB (MediaPipe yêu cầu RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phát hiện khuôn mặt trên ảnh giảm kích thước
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.4, fy=0.4)
        detected_faces = self.model.detector.detect_faces(small_frame)
        face_found = len(detected_faces)

        if face_found > 1:
            frame = AddVietnameseText.add_vietnamese_text(frame, "Có nhiều hơn 1 khuôn mặt", (10, 30))
        elif face_found == 0:
            frame = AddVietnameseText.add_vietnamese_text(frame, "Không phát hiện khuôn mặt", (10, 30))
            self.person_detected["UNKNOWN"] += 1
            self.id_arr.append("UNKNOWN")
            detected_id = "UNKNOWN"
        else:
            x, y, width, height = detected_faces[0]['box']

            # Giữ nguyên logic cũ: scale theo 1/0.4 vì detect trên ảnh 0.4x
            scale_factor = 1 / 0.4
            x, y = max(0, int(x * scale_factor)), max(0, int(y * scale_factor))
            width, height = int(width * scale_factor), int(height * scale_factor)

            # Bỏ qua khuôn mặt quá nhỏ (giữ nguyên cách tính cũ)
            face_height_ratio = height / small_frame.shape[0]
            if face_height_ratio > MIN_FACE_HEIGHT_RATIO:
                # Cắt & resize từ khung RGB gốc
                cropped = rgb_frame[y: y + height, x: x + width]
                # bảo vệ chỉ số nếu bbox chạm biên
                if cropped.size == 0:
                    return frame, None
                scaled = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
                chuan_hoa = facenet.chuan_hoa_anh(scaled)
                reshaped = chuan_hoa.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

                # Tính embedding
                feed_dict = {
                    self.model.images_placeholder: reshaped,
                    self.model.phase_train_placeholder: False
                }
                emb_arr = self.model.sess.run(self.model.embeddings, feed_dict=feed_dict)

                # Nhận diện
                predictions = self.model.model.predict_proba(emb_arr)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                best_name = self.model.class_names[best_class_indices[0]]
                print("ID: {}, Độ chính xác: {}".format(best_name, best_class_probabilities))

                if best_class_probabilities >= 0.9:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    frame = AddVietnameseText.add_vietnamese_text(frame, best_name, (x, y + height + 10), 24, (0, 255, 0))
                    frame = AddVietnameseText.add_vietnamese_text(frame, str(round(best_class_probabilities[0], 3)), (x, y + height + 30), 24, (0, 255, 0))
                    self.person_detected[best_name] += 1
                    self.id_arr.append(int(best_name))
                    detected_id = int(best_name)
                    self.last_face_location = (x, y, width, height)
                    self.last_face_id = best_name
                    self.face_ttl = 5
                else:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                    frame = AddVietnameseText.add_vietnamese_text(frame, "UNKNOWN", (x, y + height + 30), 24, (255, 0, 0))
                    self.person_detected["UNKNOWN"] += 1
                    self.id_arr.append("UNKNOWN")
                    detected_id = "UNKNOWN"
        return frame, detected_id

    def get_most_common_id(self):
        """Trả về ID xuất hiện nhiều nhất trong các frame đã xử lý"""
        if not self.id_arr:
            return "UNKNOWN"
        counter = Counter(self.id_arr)
        id_employee = counter.most_common(1)[0][0]
        return id_employee

    def reset(self):
        """Reset lại các biến theo dõi"""
        self.id_arr = []
        self.person_detected = collections.Counter()

def main():
    camera = cv2.VideoCapture(0)
    face_rec_cam = FaceRecognitionCam()
    face_rec_cam.check_model_loaded()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Không thể đọc frame từ camera")
            break

        processed_frame, current_id = face_rec_cam.process_frame(frame)
        processed_frame = imutils.resize(processed_frame, width=640)
        cv2.imshow("Face Recognition", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
