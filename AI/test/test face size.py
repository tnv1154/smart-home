
import cv2
from mtcnn import MTCNN

# Khởi tạo detector MTCNN
detector = MTCNN()

# Đọc ảnh từ webcam (camera 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
    # Chuyển ảnh sang RGB (MTCNN cần RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        # Lấy bounding box: [x, y, width, height]
        x, y, w, h = face['box']
        confidence = face['confidence']

        # Vẽ khung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Ghi thông tin vị trí & kích thước
        info = f"Pos:({x},{y}) Size:({w}x{h}) Conf:{confidence:.2f}"
        cv2.putText(frame, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Hiển thị ảnh
    cv2.imshow("MTCNN Face Detection", frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
