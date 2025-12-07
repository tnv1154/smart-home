import cv2
import numpy as np
import mediapipe as mp   # ✅ dùng MediaPipe

# ======= Thay class & khởi tạo theo MediaPipe =======
class MediaPipeFaceDetector:
    """
    Trả format giống MTCNN:
      [{'box':[x,y,w,h],
        'keypoints': {'left_eye':(x,y),'right_eye':(x,y),'nose':(x,y)},
        'confidence': float}]
    """
    def __init__(self, min_conf=0.5, long_range=False):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1 if long_range else 0,
            min_detection_confidence=min_conf
        )

    def detect_faces(self, image_bgr):
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)

        out = []
        if not res.detections:
            return out

        for d in res.detections:
            r = d.location_data.relative_bounding_box
            # clip 0..1 rồi đổi pixel
            xmin = max(0.0, r.xmin); ymin = max(0.0, r.ymin)
            xmax = min(1.0, r.xmin + r.width)
            ymax = min(1.0, r.ymin + r.height)
            x1 = int(xmin * w); y1 = int(ymin * h)
            x2 = int(xmax * w); y2 = int(ymax * h)
            bw = max(1, x2 - x1); bh = max(1, y2 - y1)

            # Thứ tự keypoint MP: 0=right_eye, 1=left_eye, 2=nose_tip
            kps = d.location_data.relative_keypoints
            kp_dict = {
                'left_eye' : (int(kps[1].x * w), int(kps[1].y * h)),
                'right_eye': (int(kps[0].x * w), int(kps[0].y * h)),
                'nose'     : (int(kps[2].x * w), int(kps[2].y * h)),
            }
            conf = float(d.score[0]) if d.score else 0.0
            out.append({'box': [x1, y1, bw, bh], 'keypoints': kp_dict, 'confidence': conf})
        return out

    def __del__(self):
        try:
            self.detector.close()
        except Exception:
            pass
# ====================================================


class FaceOrientation:
    def calculate_Angle(self, x, y, z):
        """Tính góc tạo bởi 3 điểm a, b, c"""
        a = np.array(x, dtype=np.float32)
        b = np.array(y, dtype=np.float32)
        c = np.array(z, dtype=np.float32)
        ba = a - b
        bc = c - b
        tich_2_vector = np.dot(ba, bc)
        len_ba = np.linalg.norm(ba)
        len_bc = np.linalg.norm(bc)
        cos = tich_2_vector / (len_ba * len_bc)
        goc_rad = np.arccos(cos)
        return np.degrees(goc_rad)

    def face_orientation_detection(self, detected_faces):
        """Phát hiện góc của khuôn mặt"""
        left_eye = detected_faces[0]['keypoints']['left_eye']
        right_eye = detected_faces[0]['keypoints']['right_eye']
        nose = detected_faces[0]['keypoints']['nose']

        goc_trai = self.calculate_Angle(right_eye, left_eye, nose)
        goc_phai = self.calculate_Angle(left_eye, right_eye, nose)

        hieu = abs(goc_trai - goc_phai)

        print(f"Goc trai: {goc_trai}")
        print(f"Goc phai: {goc_phai}")
        print(f"Hieu: {hieu}")
        print()

        if int(hieu) < 20:
            if int(goc_trai) in range(40, 55) and int(goc_phai) in range(40, 55):
                face_orientation = "Front"
            else:
                if int(goc_trai) <= 50 and int(goc_phai) <= 50:
                    face_orientation = "Ngua"
                else:
                    face_orientation = "cui"
        elif int(hieu) >= 60:
            if goc_phai < goc_trai:
                face_orientation = "Ngua Right"
            else:
                face_orientation = "Ngua Left"
        else:
            if goc_phai < goc_trai:
                face_orientation = "Right"
            else:
                face_orientation = "Left"
        print(face_orientation)
        return face_orientation


def main():
    cap = cv2.VideoCapture(0)
    # 🔄 ĐỔI CÁCH KHỞI TẠO: dùng MediaPipeFaceDetector
    detector = MediaPipeFaceDetector(min_conf=0.5, long_range=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

        detected_faces = detector.detect_faces(small_frame)
        face_found = len(detected_faces)
        if face_found == 1:
            x, y, width, height = detected_faces[0]['box']
            cv2.rectangle(small_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            face_orientation = FaceOrientation().face_orientation_detection(detected_faces)
            cv2.putText(small_frame, face_orientation, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            left_eye = detected_faces[0]['keypoints']['left_eye']
            right_eye = detected_faces[0]['keypoints']['right_eye']
            nose = detected_faces[0]['keypoints']['nose']
            cv2.circle(small_frame, (left_eye[0], left_eye[1]), 2, (0, 0, 255), -1)
            cv2.circle(small_frame, (right_eye[0], right_eye[1]), 2, (0, 0, 255), -1)
            cv2.circle(small_frame, (nose[0], nose[1]), 2, (0, 0, 255), -1)

            cv2.line(small_frame, left_eye, right_eye, (0, 255, 0), 2)
            cv2.line(small_frame, right_eye, nose, (0, 255, 0), 2)
            cv2.line(small_frame, nose, left_eye, (0, 255, 0), 2)

        cv2.imshow("Face Detection", small_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
