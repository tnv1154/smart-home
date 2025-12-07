import cv2
import mediapipe as mp


def _mp_bbox_to_xywh(det, img_w, img_h):
    """Chuyển bbox chuẩn hoá của MediaPipe về (x, y, w, h) pixel + cắt biên an toàn."""
    r = det.location_data.relative_bounding_box
    xmin = max(0.0, r.xmin)
    ymin = max(0.0, r.ymin)
    xmax = min(1.0, r.xmin + r.width)
    ymax = min(1.0, r.ymin + r.height)
    x1 = int(xmin * img_w)
    y1 = int(ymin * img_h)
    x2 = int(xmax * img_w)
    y2 = int(ymax * img_h)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _draw_rect(img, rect, color, thickness=2):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def _make_rect_around_ellipse(center, a, b, scale, img_w, img_h):
    """Tạo hình chữ nhật cùng tâm với elip, to/nhỏ theo scale."""
    cx, cy = center
    rw = int(2 * a * scale)
    rh = int(2 * b * scale)

    rx = int(cx - rw / 2)
    ry = int(cy - rh / 2)

    # cắt biên cho an toàn
    rx = max(0, rx)
    ry = max(0, ry)
    if rx + rw > img_w:
        rw = img_w - rx
    if ry + rh > img_h:
        rh = img_h - ry

    return (rx, ry, rw, rh)


def _box_inside(box, container):
    bx, by, bw, bh = box
    cx, cy, cw, ch = container
    return (
        bx >= cx and
        by >= cy and
        bx + bw <= cx + cw and
        by + bh <= cy + ch
    )


def check_face_in_ellipse(frame, face_box):
    """
    - Vẽ elip dọc giữa màn hình (giữ như cũ).
    - Tạo:
        + outer_rect: hình chữ nhật TO bao quanh elip (có thể to hơn xíu).
        + inner_rect: hình chữ nhật NHỎ nằm bên trong elip.
    - Điều kiện OK: bbox nằm TRỌN trong outer_rect và KHÔNG nằm trọn trong inner_rect.
    """
    h, w = frame.shape[:2]

    # ===== ELIP DỌC (GIỮ NGUYÊN) =====
    center_ellipse = (w // 2, int(h * 0.45))
    a = int(w * 0.14)        # bán trục ngang
    b = int(h * 0.35)        # bán trục dọc
    axes_ellipse = (a, b)

    # ===== RECT GỐC BAO ELIP (BẰNG ĐÚNG KÍCH THƯỚC ELIP) =====
    base_w = 2 * a
    base_h = 2 * b
    base_x = center_ellipse[0] - a
    base_y = center_ellipse[1] - b

    # ===== HÌNH CHỮ NHẬT TO (OUTER) – BAO QUANH ELIP, TO HƠN MỘT CHÚT =====
    scale_outer = 1.15   # tăng/giảm tuỳ bạn
    ow = int(base_w * scale_outer)
    oh = int(base_h * scale_outer)
    ox = int(center_ellipse[0] - ow / 2)
    oy = int(center_ellipse[1] - oh / 2)

    # cắt biên
    ox = max(0, ox)
    oy = max(0, oy)
    if ox + ow > w:
        ow = w - ox
    if oy + oh > h:
        oh = h - oy

    outer_rect = (ox, oy, ow, oh)

    # ===== HÌNH CHỮ NHẬT NHỎ (INNER) – NẰM BÊN TRONG ELIP =====
    scale_inner = 0.9    # nhỏ hơn 1 -> nhỏ hơn elip
    iw = int(base_w * scale_inner)
    ih = int(base_h * scale_inner)
    ix = int(center_ellipse[0] - iw / 2)
    iy = int(center_ellipse[1] - ih / 2)

    ix = max(0, ix)
    iy = max(0, iy)
    if ix + iw > w:
        iw = w - ix
    if iy + ih > h:
        ih = h - iy

    inner_rect = (ix, iy, iw, ih)

    # ===== KHÔNG CÓ KHUÔN MẶT =====
    if face_box is None:
        return "Khong thay khuon mat", (0, 0, 255), inner_rect, outer_rect, center_ellipse, axes_ellipse

    # ===== LOGIC MÀ BẠN MUỐN =====
    inside_outer = _box_inside(face_box, outer_rect)
    inside_inner = _box_inside(face_box, inner_rect)

    if inside_outer and not inside_inner:
        status = "OK"   # trong outer, ngoài inner
        color = (0, 255, 0)
    elif not inside_outer:
        status = "GAN"
        color = (0, 0, 255)
    else:  # inside_inner == True
        status = "XA"
        color = (0, 165, 255)  # cam để dễ phân biệt
    return status, color, inner_rect, outer_rect, center_ellipse, axes_ellipse


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_fd = mp.solutions.face_detection

    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # frame = cv2.flip(frame, 1)  # nếu muốn kiểu gương thì bật dòng này

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)

            face_box = None
            if res.detections:
                best = None
                best_area = 0
                for d in res.detections:
                    rect = _mp_bbox_to_xywh(d, w, h)
                    area = rect[2] * rect[3]
                    if area > best_area:
                        best_area = area
                        best = rect
                face_box = best

            # kiểm tra theo 2 hình chữ nhật + elip
            status, color, inner_rect, outer_rect, center_ellipse, axes_ellipse = check_face_in_ellipse(
                frame, face_box
            )

            # vẽ elip (chỉ để căn mặt cho đẹp)
            cv2.ellipse(frame, center_ellipse, axes_ellipse, 0, 0, 360, (220, 220, 220), 5)

            # vẽ 2 hình chữ nhật
            _draw_rect(frame, outer_rect, (255, 255, 255), 1)   # ngoài (trắng)
            _draw_rect(frame, inner_rect, (0, 255, 255), 1)     # trong (vàng/cyan)

            # vẽ bbox khuôn mặt
            if face_box is not None:
                _draw_rect(frame, face_box, color, 2)

            # text trạng thái
            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )

            cv2.imshow("Face check with 2 rectangles & ellipse (MediaPipe)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
