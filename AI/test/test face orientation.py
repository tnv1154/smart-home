import imutils
from mtcnn import MTCNN
import cv2
from AI.src.facenet import chuan_hoa_anh, to_rgb
import numpy as np

def calculate_distance_point(x, y, z, t):
    """TÍnh khoảng caách giữa 2 vector xy, zt"""
    a = np.array(x, dtype=np.float32)
    b = np.array(y, dtype=np.float32)
    c = np.array(z, dtype=np.float32)
    d = np.array(t, dtype=np.float32)
    mid_eye = (a + b) / 2
    mid_mouth = (c + d) / 2

    vector = mid_mouth - mid_eye
    distance = np.arctan2(vector[1], vector[0]) * 180 / np.pi
    return distance


def calculate_Angle(x, y, z):
    """Tính góc tạo bởi 3 điểm a, b, c"""
    #chuyển list thành numpy array
    a = np.array(x, dtype=np.float32)
    b = np.array(y, dtype=np.float32)
    c = np.array(z, dtype=np.float32)
    # tạo 2 vector ba, bc
    ba = a - b
    bc = c - b
    # tích góc tạo bởi 2 vector
    tich_2_vector = np.dot(ba, bc)
    len_ba = np.linalg.norm(ba)
    len_bc = np.linalg.norm(bc)
    cos = tich_2_vector / (len_ba * len_bc)
    goc_rad = np.arccos(cos)
    return np.degrees(goc_rad)


def main():
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        #small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        #small_frame = chuan_hoa_anh(small_frame)
        detected_faces = detector.detect_faces(small_frame)
        face_found = len(detected_faces)
        if face_found == 1:
            x, y, width, height = detected_faces[0]['box']
            cv2.rectangle(small_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            left_eye = detected_faces[0]['keypoints']['left_eye']
            right_eye = detected_faces[0]['keypoints']['right_eye']
            nose = detected_faces[0]['keypoints']['nose']
            left_mouth = detected_faces[0]['keypoints']['mouth_left']
            right_mouth = detected_faces[0]['keypoints']['mouth_right']

            cv2.circle(small_frame, (left_eye[0], left_eye[1]), 2, (0, 0, 255), -1)
            cv2.circle(small_frame, (right_eye[0], right_eye[1]), 2, (0, 0, 255), -1)
            cv2.circle(small_frame, (nose[0], nose[1]), 2, (0, 0, 255), -1)

            cv2.line(small_frame, left_eye, right_eye, (0, 255, 0), 2)
            cv2.line(small_frame, right_eye, nose, (0, 255, 0), 2)
            cv2.line(small_frame, nose, left_eye, (0, 255, 0), 2)

            goc_trai = calculate_Angle(right_eye, left_eye, nose)
            goc_phai = calculate_Angle(left_eye, right_eye, nose)

            eyes_mouth_distance = calculate_distance_point(left_eye, right_eye, left_mouth, right_mouth)
            print(f"Goc : {eyes_mouth_distance}")
            print(f"face_hight : {height}")
            print(f"face_width : {width}")

            cv2.putText(small_frame, "Distance: " + str(eyes_mouth_distance), (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(goc_trai, goc_phai)
            if int(goc_phai) in range(40, 60) and int(goc_trai) in range(40, 60):
                face_orientation = "Front"
            else:
                if goc_phai < goc_trai:
                    face_orientation = "Left"
                else:
                    face_orientation = "Right"
            cv2.putText(small_frame, face_orientation, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(small_frame, "goc_trai: " + str(goc_trai) + "°", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(small_frame, "goc_phai: " + str(goc_phai) + "°", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Face Detection", small_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()