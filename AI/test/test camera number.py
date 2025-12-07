import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def probe_camera(index, timeout=1.0):
    """Trả về tuple (index, active, info)
    active True nếu mở và đọc frame thành công"""
    start = time.time()
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(index)
    if not cap.isOpened():
        return index, False, "cannot open"
    # thử đọc frame tới khi timeout
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    success = False
    info = []
    while time.time() - start < timeout:
        ret, frame = cap.read()
        if ret and frame is not None:
            success = True
            # lấy một vài thông tin cơ bản
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            info = [f"resolution={w}x{h}", f"fps={fps:.1f}"]
            break
        time.sleep(0.01)
    cap.release()
    if success:
        return index, True, ";".join(info) if info else "ok"
    return index, False, "no frame"

def scan_cameras(max_index=10, timeout=1.0, workers=8):
    """Quét song song các index và trả về danh sách index đang hoạt động"""
    active = []
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(probe_camera, i, timeout): i for i in range(max_index)}
        for fut in as_completed(futures):
            idx, is_active, info = fut.result()
            results[idx] = (is_active, info)
            if is_active:
                active.append(idx)
    # sắp xếp kết quả theo index
    active.sort()
    return active, results

def main():
    max_index = 12      # thay đổi tuỳ nhu cầu
    timeout = 1.0       # giây cho mỗi camera
    workers = 8
    print(f"Scanning camera indices 0..{max_index-1} with timeout {timeout}s ...")
    active, results = scan_cameras(max_index=max_index, timeout=timeout, workers=workers)
    if active:
        print("Active camera indices found:", active)
        for i in active:
            print(f" - {i}: {results[i][1]}")
    else:
        print("No active cameras detected.")
    # in toàn bộ kết quả chi tiết
    print("\nFull probe details:")
    for i in range(max_index):
        status, info = results.get(i, (False, "not tested"))
        print(f"{i:02d}: {'ACTIVE' if status else 'INACTIVE'} - {info}")

if __name__ == "__main__":
    main()
