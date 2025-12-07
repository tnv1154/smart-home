from flask import Flask, render_template, jsonify, request, session, redirect, url_for

import requests
from datetime import datetime
import json
from functools import wraps
import mysql.connector
from mysql.connector import Error, pooling
import hashlib

import os
import time
import base64
import threading
from collections import deque

import cv2
import numpy as np
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
app.secret_key = 'your-secret-key-change-this-in-production'

# ===== CẤU HÌNH DATABASE =====
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'iot'
}

# ===== KẾT NỐI DB DÙNG POOL =====
try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="iot_pool",
        pool_size=5,          # tuỳ mức tải, 5–10 là đủ
        **DB_CONFIG
    )
    print("✅ MySQL connection pool created")
except Error as e:
    print(f"❌ Error creating MySQL pool: {e}")
    db_pool = None


# ===== CẬP NHẬT ĐỊA CHỈ IP ESP32 CỦA BẠN =====
ESP32_IP = "10.242.89.41"
ESP32_BASE_URL = f"http://{ESP32_IP}"

# Lưu trữ tên chủ thẻ và lịch sử
card_names = {}
access_history = deque(maxlen=500)  # chỉ giữ 500 bản ghi mới nhất


# ===== FOLDER LƯU ẢNH TRUY CẬP =====
ACCESS_LOGS_FOLDER = "access_logs"
os.makedirs(ACCESS_LOGS_FOLDER, exist_ok=True)

# ================== FACE RECOGNITION CONFIG ==================
os.makedirs(RAW_FOLDER, exist_ok=True)

face_rec_cam = FaceRecognitionCam()
face_rec_cam.check_model_loaded()

add_sessions = {}

# ===== CAMERA SHARED FRAME =====
camera = None
camera_lock = threading.Lock()
background_using_camera = False
camera_initialized = False

# Frame buffer để chia sẻ giữa Security Camera và Background thread
latest_frame = None
latest_frame_time = 0
frame_lock = threading.Lock()

face_rec_lock = threading.Lock()

# ===== TELEGRAM CONFIG =====
TELEGRAM_TOKEN = "7850063944:AAHoZeCVGu2PuRswtzKWqhwm3WuuGlzlbEg"      # copy từ send_image_and_mes.py
TELEGRAM_CHAT_ID = "6717680448"    # id người nhận /yes /no
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
TELEGRAM_LAST_UPDATE_ID = 0


def telegram_sync_offset():
    """
    Đọc hết các update cũ trong hàng đợi và chỉ lưu lại update_id cuối cùng,
    KHÔNG xử lý /yes /no ở đây.
    Mục tiêu: tránh việc lần chờ sau vẫn ăn lại các lệnh cũ.
    """
    global TELEGRAM_LAST_UPDATE_ID
    try:
        resp = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={
                "offset": TELEGRAM_LAST_UPDATE_ID + 1,
                "timeout": 1
            },
            timeout=3
        )
        data = resp.json()
        results = data.get("result", [])
        if results:
            TELEGRAM_LAST_UPDATE_ID = results[-1]["update_id"]
            print(f"Telegram sync_offset: last_update_id = {TELEGRAM_LAST_UPDATE_ID}")
    except Exception as e:
        print("Telegram sync_offset error:", e)



def telegram_send_photo_and_message(image_path, message_text):
    try:
        url_photo = f"{TELEGRAM_API}/sendPhoto"
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": message_text
            }
            r = requests.post(url_photo, data=data, files=files, timeout=10)
        print("Telegram sendPhoto result:", r.json())
    except Exception as e:
        print("Telegram sendPhoto error:", e)


def telegram_send_message(message_text):
    try:
        url_message = f"{TELEGRAM_API}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message_text}
        r = requests.post(url_message, data=payload, timeout=10)
        print("Telegram sendMessage result:", r.json())
    except Exception as e:
        print("Telegram sendMessage error:", e)


def telegram_wait_for_decision(timeout_window=30):
    """
    Chờ user gửi /yes hoặc /no trong khoảng timeout_window (giây).
    Trả về: "yes", "no" hoặc None (hết giờ không nhận được gì).
    """
    global TELEGRAM_LAST_UPDATE_ID

    deadline = time.time() + timeout_window
    decision = None

    while time.time() < deadline and decision is None:
        try:
            resp = requests.get(
                f"{TELEGRAM_API}/getUpdates",
                params={
                    "offset": TELEGRAM_LAST_UPDATE_ID + 1,
                    "timeout": 5
                },
                timeout=7
            )
            data = resp.json()

            for update in data.get("result", []):
                TELEGRAM_LAST_UPDATE_ID = update["update_id"]

                message = update.get("message")
                if not message:
                    continue

                # CHỈ xử lý tin nhắn từ đúng chat_id của bạn
                chat = message.get("chat") or {}
                chat_id = str(chat.get("id", ""))
                if chat_id != str(TELEGRAM_CHAT_ID):
                    continue

                text = (message.get("text") or "").strip().lower()

                if text == "/yes":
                    decision = "yes"
                    print("Telegram: nhận /yes")
                elif text == "/no":
                    decision = "no"
                    print("Telegram: nhận /no")

            if decision is not None:
                break

        except Exception as e:
            print("Telegram getUpdates error:", e)
            break

    return decision




def get_camera():
    """Lấy camera instance (singleton pattern) - CHỈ cho Add Face"""
    global camera, camera_initialized

    if camera is None or not camera.isOpened():
        print("📹 Đang khởi tạo camera...")
        try:
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera_initialized = True
                print("✅ Camera khởi tạo thành công!")
            else:
                print("❌ Không thể mở camera!")
                camera = None
        except Exception as e:
            print(f"❌ Lỗi khởi tạo camera: {e}")
            camera = None

    return camera


def update_shared_frame(frame):
    """Cập nhật frame từ Security Camera tab để Background thread dùng"""
    global latest_frame, latest_frame_time
    with frame_lock:
        latest_frame = frame.copy()
        latest_frame_time = time.time()


def get_shared_frame():
    """Lấy frame mới nhất từ Security Camera tab"""
    global latest_frame, latest_frame_time
    with frame_lock:
        if latest_frame is None:
            return None, 0
        # Trả về frame nếu còn mới (trong vòng 2 giây)
        age = time.time() - latest_frame_time
        if age > 2.0:
            return None, age
        return latest_frame.copy(), age

def add_access_log(entry: dict):
    """
    entry dạng:
    {
        'time': 'YYYY-MM-DD HH:MM:SS',
        'method': 'RFID + Face Recognition',
        'uid': '12345678',
        'name': 'Person_01',
        'result': 'Success',
        'image': 'file.jpg' (có thể None)
    }
    Lưu vào access_history (RAM) và bảng access_logs (MySQL).
    """
    # 1. Lưu RAM như cũ để không phá vỡ các chức năng hiện có
    access_history.append(entry)

    # 2. Lưu vào MySQL
    try:
        conn = get_db_connection()
        if not conn:
            return

        cursor = conn.cursor()
        sql = """
            INSERT INTO access_logs (event_time, method, uid, name, result, image)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            entry.get('time'),
            entry.get('method'),
            entry.get('uid'),
            entry.get('name'),
            entry.get('result'),
            entry.get('image'),
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"⚠️ Lỗi ghi access_log vào DB: {e}")


# ===== BACKGROUND FACE DETECTION THREAD =====
background_thread = None
background_running = False


def background_face_detection():
    """
    Background thread:

    1. ESP32 quét thẻ RFID → /api/face/status trả về {waiting: true, uid: "..."}
    2. Server lấy frame từ tab Security Camera → nhận diện khuôn mặt
    3. Logic mới:

       - Nếu nhận diện được khuôn mặt ĐÃ ĐĂNG KÝ:
           -> Lưu ảnh log
           -> Gửi detected=True, person_id=<ID khuôn mặt> về ESP32 (mở cửa)
           -> (Tuỳ chọn) Gửi thông báo Text lên Telegram

       - Nếu CÓ KHUÔN MẶT nhưng là Unknown:
           -> Lưu ảnh
           -> Gửi ảnh + thông báo lên Telegram, yêu cầu /yes hoặc /no trong 30s
           -> Nếu /yes: gửi detected=True về ESP32 (mở cửa)
           -> Nếu /no hoặc hết giờ: gửi detected=False về ESP32

       - Nếu KHÔNG PHÁT HIỆN KHUÔN MẶT:
           -> Gửi thông báo text lên Telegram (không có ảnh)
           -> Xử lý /yes /no tương tự
    """
    global background_running, background_using_camera
    print("🎥 Background Face Detection Thread started (LOGIC MỚI)!")
    print(f"📡 Polling ESP32 at: {ESP32_BASE_URL}")
    print("⚙️  Các trường hợp:")
    print("    1) Khuôn mặt đã đăng ký → MỞ CỬA luôn (auto).")
    print("    2) Khuôn mặt lạ (Unknown) → Gửi ảnh + chờ /yes /no trong 30s.")
    print("    3) Không có khuôn mặt → Gửi cảnh báo + chờ /yes /no trong 30s.")

    last_uid_processed = ""

    while background_running:
        try:
            # 1. Hỏi ESP32 xem có đang chờ face detection không
            response = requests.get(f"{ESP32_BASE_URL}/api/face/status", timeout=2)
            status = response.json()

            waiting = status.get("waiting", False)
            pending_uid = status.get("uid", "")

            # In log mỗi 5s cho dễ theo dõi
            if not hasattr(background_face_detection, 'last_print'):
                background_face_detection.last_print = 0
            if time.time() - background_face_detection.last_print > 5:
                print(f"📊 ESP32 Status: waiting={waiting}, uid={pending_uid}")
                background_face_detection.last_print = time.time()

            # Chỉ xử lý khi ESP32 báo đang chờ & UID mới
            if waiting and pending_uid and pending_uid != last_uid_processed:
                print("=" * 60)
                print(f"🔔 ESP32 yêu cầu xác thực khuôn mặt cho UID: {pending_uid}")
                print("🎬 Bắt đầu lấy frame từ Security Camera trong 10 giây...")
                print("=" * 60)

                last_uid_processed = pending_uid
                background_using_camera = True

                # Biến trạng thái trong phiên xử lý
                start_time = time.time()
                frame_count = 0
                no_frame_count = 0

                # Trạng thái nhận diện
                recognized_id = None          # ID khuôn mặt đã đăng ký (nếu có)
                recognized_count = 0          # số frame liên tiếp nhận diện được ID đó
                best_frame_known = None       # frame đẹp nhất của khuôn mặt đã đăng ký

                saw_any_face = False          # có thấy khuôn mặt (kể cả Unknown) hay không
                best_frame_unknown = None     # frame có khuôn mặt nhưng là Unknown

                print("📸 Đang lấy frame từ Security Camera tab...")
                print("💡 Hãy đảm bảo tab Security Camera đang MỞ!")

                # 2. Lặp tối đa 10 giây để lấy frame & nhận diện
                while time.time() - start_time < 10.0:
                    # Lấy frame mới nhất từ shared buffer
                    frame, frame_age = get_shared_frame()

                    if frame is None:
                        no_frame_count += 1
                        if no_frame_count % 20 == 1:
                            print(f"⚠️ Chưa có frame từ Security Camera "
                                  f"(đã chờ ~{no_frame_count * 0.1:.1f}s)")
                            print("   💡 Mở tab Security Camera để hệ thống hoạt động.")
                        time.sleep(0.1)
                        continue

                    # Lần đầu nhận được frame sau khi bị thiếu
                    if no_frame_count > 0 and frame_count == 0:
                        print(f"✅ Đã nhận frame từ Security Camera (độ trễ: {frame_age:.2f}s)")
                    no_frame_count = 0

                    frame_count += 1

                    try:
                        # Nhận diện khuôn mặt
                        processed_frame, current_id = face_rec_cam.process_frame(frame)

                        # Lấy ID ổn định (nếu có hàm get_most_common_id)
                        # DÙNG CHUNG LOCK
                        with face_rec_lock:
                            processed_frame, current_id = face_rec_cam.process_frame(frame)

                            stable_id = current_id
                            if hasattr(face_rec_cam, "get_most_common_id"):
                                try:
                                    stable_id = face_rec_cam.get_most_common_id()
                                except Exception:
                                    pass

                        norm_id = (str(stable_id) or "").strip()
                        norm_id_lower = norm_id.lower()

                        # PHÂN LOẠI:
                        # 1) Khuôn mặt đã đăng ký: ID không rỗng, không phải "unknown"/"no face"
                        if norm_id and norm_id_lower not in ("unknown", "no face", "noface"):
                            saw_any_face = True
                            recognized_id = norm_id
                            recognized_count += 1
                            best_frame_known = processed_frame.copy()
                            print(f"✅ Frame {frame_count}: Nhận diện được khuôn mặt ID = {recognized_id} "
                                  f"(đếm {recognized_count})")

                            # Nếu nhận diện đủ 3 lần → confirm luôn
                            if recognized_count >= 3:
                                print(f"🎯 XÁC NHẬN: Khuôn mặt '{recognized_id}' hợp lệ (>=3 lần).")
                                break

                        # 2) Khuôn mặt lạ (Unknown)
                        elif norm_id_lower == "unknown":
                            saw_any_face = True
                            if best_frame_unknown is None:
                                best_frame_unknown = processed_frame.copy()
                            if frame_count % 10 == 0:
                                print(f"⚠️ Frame {frame_count}: Thấy khuôn mặt nhưng là 'Unknown'.")

                        # 3) Không thấy khuôn mặt (ID rỗng / 'no face' / ...)
                        else:
                            if frame_count % 15 == 0:
                                print(f"⏳ Frame {frame_count}: Chưa phát hiện khuôn mặt hợp lệ.")

                    except Exception as e:
                        if frame_count % 10 == 1:
                            print(f"⚠️ Frame {frame_count}: Lỗi xử lý - {e}")

                    time.sleep(0.05)

                background_using_camera = False

                # 3. Tóm tắt
                duration = time.time() - start_time
                print("=" * 60)
                print(f"📋 Tóm tắt phiên xử lý UID {pending_uid}:")
                print(f"   - Thời gian xử lý: {duration:.1f}s")
                print(f"   - Số frame đọc được: {frame_count}")
                print(f"   - Nhận diện OK (ID đã đăng ký): {recognized_id} "
                      f"(số lần: {recognized_count})")
                print(f"   - Có thấy khuôn mặt (kể cả Unknown): {saw_any_face}")
                if no_frame_count > 0:
                    print(f"   ⚠️ Số lần không nhận được frame: {no_frame_count}")
                    print("      💡 Hãy giữ tab Security Camera luôn mở.")
                print("=" * 60)

                # 4. Quyết định theo 3 trường hợp

                # Thời gian / filename chung
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # ==== TRƯỜNG HỢP 1: Khuôn mặt đã đăng ký → MỞ CỬA NGAY ====
                if recognized_id and recognized_count >= 1 and best_frame_known is not None:
                    print(f"✅ Kết luận: UID {pending_uid} đi kèm khuôn mặt ĐÃ ĐĂNG KÝ ({recognized_id}).")
                    print("   → MỞ CỬA TỰ ĐỘNG (không cần xác nhận Telegram).")

                    # Lưu ảnh log
                    filename = f"{pending_uid}_OK_{recognized_id}_{timestamp}.jpg"
                    filepath = os.path.join(ACCESS_LOGS_FOLDER, filename)
                    cv2.imwrite(filepath, best_frame_known)
                    print(f"💾 Đã lưu ảnh: {filepath}")

                    # Gửi kết quả về ESP32
                    payload = {"detected": True, "person_id": recognized_id}
                    print(f"📤 Gửi về ESP32 /api/face/detected: {payload}")
                    try:
                        result = requests.post(
                            f"{ESP32_BASE_URL}/api/face/detected",
                            json=payload,
                            timeout=2
                        )
                        print(f"📥 ESP32 phản hồi: {result.json()}")
                    except Exception as e:
                        print(f"⚠️ Lỗi gửi kết quả về ESP32: {e}")

                    # Ghi lịch sử
                    add_access_log({
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'method': 'RFID + Face Recognition',
                        'uid': pending_uid,
                        'name': recognized_id,
                        'result': 'Success',
                        'image': filename
                    })

                    # (Tuỳ chọn) Gửi thông báo text lên Telegram
                    try:
                        msg = (
                            "✅ Truy cập hợp lệ\n"
                            f"- UID thẻ: {pending_uid}\n"
                            f"- Người dùng: {recognized_id}\n"
                            f"- Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        telegram_send_message(msg)
                    except Exception as e:
                        print(f"⚠️ Không gửi được thông báo Telegram (success): {e}")

                # ==== TRƯỜNG HỢP 2: Có khuôn mặt nhưng là UNKNOWN ====
                elif saw_any_face and best_frame_unknown is not None:
                    print(f"❌ Kết luận: UID {pending_uid} đi kèm KHUÔN MẶT LẠ (Unknown).")
                    print("   → Gửi ảnh + yêu cầu xác nhận /yes hoặc /no trong 30s.")

                    # Lưu ảnh để gửi lên Telegram
                    filename = f"{pending_uid}_UNKNOWN_{timestamp}.jpg"
                    filepath = os.path.join(ACCESS_LOGS_FOLDER, filename)
                    cv2.imwrite(filepath, best_frame_unknown)
                    print(f"💾 Đã lưu ảnh Unknown: {filepath}")

                    # DỌN SẠCH UPDATE CŨ TRƯỚC KHI HỎI Ý KIẾN
                    telegram_sync_offset()

                    # Gửi ảnh + tin nhắn lên Telegram
                    message_text = (
                        "⚠️ CẢNH BÁO: Có người quét thẻ nhưng khuôn mặt KHÔNG HỢP LỆ hoặc CHƯA ĐĂNG KÝ.\n"
                        f"- UID thẻ: {pending_uid}\n"
                        "Nếu muốn MỞ CỬA, hãy trả lời /yes trong vòng 30 giây.\n"
                        "Nếu muốn TỪ CHỐI, hãy trả lời /no hoặc bỏ qua."
                    )
                    telegram_send_photo_and_message(filepath, message_text)

                    # Chờ /yes hoặc /no trong 30s
                    decision = telegram_wait_for_decision(timeout_window=30)
                    print(f"📨 Quyết định Telegram cho UID {pending_uid}: {decision}")

                    if decision == "yes":
                        print("✅ Admin gửi /yes → MỞ CỬA (Override).")
                        payload = {"detected": True, "person_id": "ManualApproved"}
                        try:
                            result = requests.post(
                                f"{ESP32_BASE_URL}/api/face/detected",
                                json=payload,
                                timeout=2
                            )
                            print(f"📥 ESP32 phản hồi: {result.json()}")
                        except Exception as e:
                            print(f"⚠️ Lỗi gửi kết quả về ESP32: {e}")

                        telegram_send_message(
                            f"✅ Đã MỞ CỬA theo yêu cầu (/yes) cho UID {pending_uid}."
                        )

                        add_access_log({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'method': 'RFID + Face Unknown + Telegram /yes',
                            'uid': pending_uid,
                            'name': 'Unknown (Approved)',
                            'result': 'Manual Approved',
                            'image': filename
                        })


                    elif decision == "no":
                        print("⛔ Admin gửi /no → KHÔNG MỞ CỬA.")
                        payload = {"detected": False, "person_id": "Rejected"}
                        try:
                            result = requests.post(
                                f"{ESP32_BASE_URL}/api/face/detected",
                                json=payload,
                                timeout=2
                            )
                            print(f"📥 ESP32 phản hồi: {result.json()}")
                        except Exception as e:
                            print(f"⚠️ Lỗi gửi kết quả về ESP32: {e}")

                        telegram_send_message(
                            f"⛔ ĐÃ TỪ CHỐI mở cửa (/no) cho UID {pending_uid}."
                        )

                        add_access_log({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'method': 'RFID + Face Unknown + Telegram /no',
                            'uid': pending_uid,
                            'name': 'Unknown (Rejected)',
                            'result': 'Rejected',
                            'image': filename
                        })

                    else:
                        print("⏰ Hết thời gian chờ, không nhận được /yes hoặc /no → KHÔNG MỞ CỬA.")
                        payload = {"detected": False, "person_id": "Timeout"}
                        try:
                            result = requests.post(
                                f"{ESP32_BASE_URL}/api/face/detected",
                                json=payload,
                                timeout=2
                            )
                            print(f"📥 ESP32 phản hồi: {result.json()}")
                        except Exception as e:
                            print(f"⚠️ Lỗi gửi kết quả về ESP32: {e}")

                        telegram_send_message(
                            f"⏰ Hết thời gian 30s, KHÔNG mở cửa cho UID {pending_uid}."
                        )

                        add_access_log({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'method': 'RFID + Face Unknown + Timeout',
                            'uid': pending_uid,
                            'name': 'Unknown (Timeout)',
                            'result': 'Timeout',
                            'image': filename
                        })

                # ==== TRƯỜNG HỢP 3: HOÀN TOÀN KHÔNG CÓ KHUÔN MẶT ====
                else:
                    print(f"❌ Kết luận: UID {pending_uid} nhưng KHÔNG phát hiện khuôn mặt nào trước cửa.")
                    print("   → Gửi cảnh báo text + chờ /yes /no trong 30s.")



                    message_text = (
                        "⚠️ CẢNH BÁO: Có người quét thẻ nhưng KHÔNG phát hiện khuôn mặt trước cửa.\n"
                        f"- UID thẻ: {pending_uid}\n"
                        "Nếu vẫn muốn MỞ CỬA, hãy trả lời /yes trong vòng 30 giây.\n"
                        "Nếu muốn TỪ CHỐI, hãy trả lời /no hoặc bỏ qua."
                    )
                    # DỌN SẠCH UPDATE CŨ TRƯỚC KHI HỎI Ý KIẾN
                    telegram_sync_offset()

                    telegram_send_message(message_text)

                    decision = telegram_wait_for_decision(timeout_window=30)
                    print(f"📨 Quyết định Telegram cho UID {pending_uid} (NO FACE): {decision}")

                    if decision == "yes":
                        print("✅ Admin gửi /yes (NO FACE) → MỞ CỬA (Override).")
                        payload = {"detected": True, "person_id": "NoFace_Approved"}
                        try:
                            result = requests.post(
                                f"{ESP32_BASE_URL}/api/face/detected",
                                json=payload,
                                timeout=2
                            )
                            print(f"📥 ESP32 phản hồi: {result.json()}")
                        except Exception as e:
                            print(f"⚠️ Lỗi gửi kết quả về ESP32: {e}")

                        telegram_send_message(
                            f"✅ Đã MỞ CỬA theo yêu cầu (/yes – không có mặt) cho UID {pending_uid}."
                        )

                        add_access_log({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'method': 'RFID + No Face + Telegram /yes',
                            'uid': pending_uid,
                            'name': 'NoFace (Approved)',
                            'result': 'Manual Approved'
                        })

                    elif decision == "no":
                        print("⛔ Admin gửi /no (NO FACE) → KHÔNG MỞ CỬA.")
                        payload = {"detected": False, "person_id": "NoFace_Rejected"}
                        try:
                            result = requests.post(
                                f"{ESP32_BASE_URL}/api/face/detected",
                                json=payload,
                                timeout=2
                            )
                            print(f"📥 ESP32 phản hồi: {result.json()}")
                        except Exception as e:
                            print(f"⚠️ Lỗi gửi kết quả về ESP32: {e}")

                        telegram_send_message(
                            f"⛔ ĐÃ TỪ CHỐI mở cửa (/no – không có mặt) cho UID {pending_uid}."
                        )

                        add_access_log({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'method': 'RFID + No Face + Telegram /no',
                            'uid': pending_uid,
                            'name': 'NoFace (Rejected)',
                            'result': 'Rejected'
                        })

                    else:
                        print("⏰ Hết thời gian chờ (NO FACE), KHÔNG MỞ CỬA.")
                        payload = {"detected": False, "person_id": "NoFace_Timeout"}
                        try:
                            result = requests.post(
                                f"{ESP32_BASE_URL}/api/face/detected",
                                json=payload,
                                timeout=2
                            )
                            print(f"📥 ESP32 phản hồi: {result.json()}")
                        except Exception as e:
                            print(f"⚠️ Lỗi gửi kết quả về ESP32: {e}")

                        telegram_send_message(
                            f"⏰ Hết thời gian 30s (không có mặt), KHÔNG mở cửa cho UID {pending_uid}."
                        )

                        add_access_log({
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'method': 'RFID + No Face + Timeout',
                            'uid': pending_uid,
                            'name': 'NoFace (Timeout)',
                            'result': 'Timeout'
                        })

                # Nghỉ 2s trước khi vòng lặp tiếp theo để tránh spam
                time.sleep(2.0)

            elif not waiting:
                # ESP32 không chờ nữa → reset UID đã xử lý
                last_uid_processed = ""
                background_using_camera = False

        except requests.exceptions.RequestException:
            # Lỗi network nhỏ với ESP32 thì bỏ qua, thử lại sau
            pass
        except Exception as e:
            print(f"⚠️ Background thread error: {e}")
            import traceback
            traceback.print_exc()
            background_using_camera = False

        time.sleep(0.5)

    print("🛑 Background Face Detection Thread stopped!")



# Hàm này không cần nữa vì đã tích hợp vào background_face_detection()


def start_background_thread():
    """Khởi động background thread"""
    global background_thread, background_running

    if background_thread is not None and background_thread.is_alive():
        print("⚠️ Background thread đã chạy rồi!")
        return

    background_running = True
    background_thread = threading.Thread(target=background_face_detection, daemon=True)
    background_thread.start()
    print("✅ Background thread started!")


def stop_background_thread():
    """Dừng background thread"""
    global background_running
    background_running = False
    if background_thread:
        background_thread.join(timeout=3)
    print("🛑 Background thread stopped!")


# ===== DATABASE FUNCTIONS =====
def get_db_connection():
    """Lấy 1 connection từ pool (hoặc tạo mới fallback)"""
    try:
        if db_pool:
            return db_pool.get_connection()
        # fallback nếu pool tạo thất bại
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"Database connection error: {e}")
        return None



def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def verify_user(username, password):
    connection = get_db_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        hashed_pwd = hash_password(password)

        query = "SELECT * FROM users WHERE username = %s AND password = %s"
        cursor.execute(query, (username, hashed_pwd))
        user = cursor.fetchone()

        cursor.close()
        connection.close()

        return user
    except Error as e:
        print(f"Verify user error: {e}")
        return None


# ===== DECORATORS =====
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)

    return decorated_function


# ===== HELPER FUNCTIONS =====
def decode_base64_image(data_url: str):
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
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode("utf-8")
    return "data:image/jpeg;base64," + b64


class FaceAddWebSession:
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

        try:
            self.speech.Trong_khung_start()
        except Exception:
            pass

    def get_target_orientation(self):
        if self.front_count < IMAGES_PER_ORIENTATION:
            return "Front"
        if self.left_count < IMAGES_PER_ORIENTATION:
            return "Left"
        if self.right_count < IMAGES_PER_ORIENTATION:
            return "Right"
        return None

    def process_frame(self, frame_bgr: np.ndarray):
        now = time.time()
        self.last_used = now  # cập nhật mỗi khi session được dùng
        target_ori = self.get_target_orientation()
        if target_ori is None:
            self.done = True
            msg_done = f"Đã chụp đủ {NUM_IMAGES} ảnh (10 thẳng, 10 trái, 10 phải)."
            frame_bgr = AddVietnameseText.add_vietnamese_text(
                frame_bgr, msg_done, (10, 30),
                font_size=25, font_color=(0, 255, 0)
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
                font_size=20, font_color=(0, 0, 255)
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
                font_size=20, font_color=(0, 0, 255)
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
            dx, dy, dw, dh = faces[0]["box"]
            sx = frame_bgr.shape[1] / detect_in.shape[1]
            sy = frame_bgr.shape[0] / detect_in.shape[0]
            x = int(round(dx * sx))
            y = int(round(dy * sy))
            w = int(round(dw * sx))
            h = int(round(dh * sy))

            x, y, w, h = expand_and_clip(
                x, y, w, h,
                frame_bgr.shape[1], frame_bgr.shape[0],
                margin=MARGIN
            )

            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            status_text, status_color, inner_rect, outer_rect, cen, axes = \
                check_face_in_ellipse(frame_bgr, (x, y, w, h))

            try:
                cv2.ellipse(frame_bgr, cen, axes, 0, 0, 360, (220, 220, 220), 5)
            except Exception:
                pass

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

            frame_bgr = AddVietnameseText.add_vietnamese_text(
                frame_bgr,
                f"Target: {target_ori} | Front {self.front_count}/10  Left {self.left_count}/10  Right {self.right_count}/10",
                (10, 70),
                font_size=20,
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


# ============== ROUTES ==============

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = verify_user(username, password)

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password!")

    return render_template('login.html', error=None)


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    return render_template(
        'index.html',
        username=session.get('username'),
        is_admin=(session.get('role') == 'admin')
    )


@app.route('/api/status')
@login_required
def get_status():
    try:
        response = requests.get(f"{ESP32_BASE_URL}/api/status", timeout=2)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            "devices": {"den1": False, "den2": False, "den3": False, "quat": False},
            "sensors": {"temperature": 0, "humidity": 0, "tempThreshold": 30},
            "door": {"open": False},
            "autoMode": False,
            "error": str(e)
        }), 503


@app.route('/api/control', methods=['POST'])
@login_required
def control_device():
    try:
        data = request.json
        response = requests.post(
            f"{ESP32_BASE_URL}/api/control",
            json=data,
            timeout=2
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/threshold', methods=['POST'])
@login_required
def set_threshold():
    try:
        data = request.json
        response = requests.post(
            f"{ESP32_BASE_URL}/api/threshold",
            json=data,
            timeout=2
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rfid/current')
@login_required
def rfid_current():
    try:
        response = requests.get(f"{ESP32_BASE_URL}/api/rfid/current", timeout=2)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"uid": "", "error": str(e)})


@app.route('/api/rfid/add', methods=['POST'])
@admin_required
def rfid_add():
    try:
        data = request.json
        name = data.get('name', 'Unknown')

        response = requests.post(f"{ESP32_BASE_URL}/api/rfid/add", timeout=2)
        result = response.json()

        if result.get('success'):
            card_names[result['uid']] = name

            add_access_log({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'RFID Card Added',
                'uid': result['uid'],
                'name': name,
                'result': 'Success'
            })

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rfid/list')
@login_required
def rfid_list():
    try:
        response = requests.get(f"{ESP32_BASE_URL}/api/rfid/list", timeout=2)
        data = response.json()

        for card in data['cards']:
            card['name'] = card_names.get(card['uid'], f"User {card['id'] + 1}")

        return jsonify(data)
    except Exception as e:
        return jsonify({"cards": [], "total": 0, "error": str(e)})


@app.route('/api/rfid/delete', methods=['DELETE'])
@admin_required
def rfid_delete():
    try:
        uid = request.args.get('uid')
        response = requests.delete(
            f"{ESP32_BASE_URL}/api/rfid/delete?uid={uid}",
            timeout=2
        )
        result = response.json()

        if result.get('success'):
            if uid in card_names:
                del card_names[uid]

            add_access_log({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'RFID Card Deleted',
                'uid': uid,
                'name': '-',
                'result': 'Info'
            })

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history')
@login_required
def get_history():
    # deque không hỗ trợ slice trực tiếp, nên ép về list
    hist = list(access_history)[-50:]
    return jsonify({"history": hist})


@app.route('/api/door/open', methods=['POST'])
@login_required
def door_open():
    try:
        response = requests.post(f"{ESP32_BASE_URL}/api/door/open", timeout=2)
        data = response.json()

        # Nếu ESP32 báo success -> ghi log
        if data.get('success'):
            add_access_log({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'Manual Door Control',
                'uid': '-',  # không có RFID
                'name': session.get('username', 'UnknownUser'),
                'result': 'Success',
                'image': None
            })

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/door/close', methods=['POST'])
@login_required
def door_close():
    try:
        response = requests.post(f"{ESP32_BASE_URL}/api/door/close", timeout=2)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ API FACE RECOGNITION ============

@app.route("/api/recognize", methods=["POST"])
@login_required
def api_recognize():
    """
    API cho Security Camera tab - nhận diện từ webcam stream của browser
    QUAN TRỌNG: Cập nhật frame vào shared buffer để Background thread dùng
    """
    data = request.get_json(silent=True) or {}
    img_data = data.get("image")

    frame = decode_base64_image(img_data)
    if frame is None:
        return jsonify({"ok": False, "error": "Không giải mã được ảnh"}), 400

    # CẬP NHẬT frame vào shared buffer cho background thread
    update_shared_frame(frame)

    try:
        # Xử lý nhận diện bình thường
        # DÙNG LOCK để mọi lời gọi vào model đều tuần tự
        with face_rec_lock:
            processed_frame, current_id = face_rec_cam.process_frame(frame)

            if hasattr(face_rec_cam, 'get_most_common_id'):
                stable_id = face_rec_cam.get_most_common_id()
            else:
                stable_id = current_id

        out_img = encode_image_to_base64(processed_frame)

        return jsonify({
            "ok": True,
            "current_id": str(current_id),
            "stable_id": str(stable_id),
            "image": out_img,
            "shared_with_auto_door": True  # Báo cho frontend biết frame đã được chia sẻ
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ============ API THÊM KHUÔN MẶT ============

@app.route("/api/add_face_stream", methods=["POST"])
@login_required
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

    session_obj = add_sessions.get(person_id)
    if session_obj is None:
        session_obj = FaceAddWebSession(person_id)
        add_sessions[person_id] = session_obj

    frame_out, info = session_obj.process_frame(frame)
    out_img = encode_image_to_base64(frame_out)

    # DỌN CÁC SESSION CŨ (không dùng > 10 phút)
    now = time.time()
    SESSION_TTL = 600  # 10 phút

    for sid, sess in list(add_sessions.items()):
        last_used = getattr(sess, "last_used", now)
        if now - last_used > SESSION_TTL:
            print(f"🧹 Xoá FaceAddWebSession cũ cho person_id={sid}")
            del add_sessions[sid]

    if info["done"] and not session_obj.trained:
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

        global face_rec_cam
        face_rec_cam = FaceRecognitionCam()
        face_rec_cam.check_model_loaded()

        session_obj.trained = True
        info["message"] = info["message"] + " | Đã align & huấn luyện lại classifier."

    return jsonify({
        "ok": True,
        "image": out_img,
        **info
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🏠 ESP32 Smart Home Control System with Authentication")
    print("=" * 60)
    print(f"📡 ESP32 IP Address: {ESP32_IP}")
    print(f"🌐 Flask Server: http://localhost:5000")
    print(f"🔐 Login Page: http://localhost:5000/login")
    print("=" * 60)
    print("👑 Admin: admin / admin123")
    print("👤 User1: user1 / user123")
    print("👤 User2: user2 / user123")
    print("👤 User3: user3 / user123")
    print("=" * 60)

    # Khởi động background thread
    start_background_thread()

    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        stop_background_thread()
        if camera:
            camera.release()
        cv2.destroyAllWindows()


