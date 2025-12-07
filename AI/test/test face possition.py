import cv2
from mtcnn import MTCNN

# Khởi tạo detector MTCNN
detector = MTCNN()

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Ngưỡng cho phép sai lệch (pixel)
TOLERANCE = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    # Vẽ điểm trung tâm khung hình
    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Vẽ bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Vẽ điểm trung tâm khuôn mặt
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 255), -1)

        # Kiểm tra khoảng cách từ tâm khuôn mặt đến tâm khung hình
        dx = abs(face_center_x - frame_center_x)
        dy = abs(face_center_y - frame_center_y)

        if dx <= TOLERANCE and dy <= TOLERANCE:
            status = "Face is centered"
            color = (0, 255, 0)
        else:
            status = "Face is NOT centered"
            color = (0, 0, 255)

        # Hiển thị thông tin
        cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Hiển thị ảnh
    cv2.imshow("MTCNN Face Center Check", frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
