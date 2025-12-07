import pygame
import time

# Khởi tạo pygame mixer
pygame.mixer.init()

# Load âm thanh (nên dùng WAV hoặc OGG để đảm bảo tương thích tốt)
sound = pygame.mixer.Sound("E:\PythonProjectMain\AI\Voice\Vui_long_nhin_thang.wav")  # Thay bằng đường dẫn file của bạn

count = 0
last_play_time = 0
delay_between_plays = 3  # Giây: phát âm thanh mỗi 3 giây

# Vòng lặp chính
while True:
    count += 1
    print(f"Count: {count}")

    current_time = time.time()

    # Mỗi 3 giây mới phát âm một lần
    if current_time - last_play_time >= delay_between_plays:
        sound.play()  # Phát âm thanh không chặn chương trình
        last_play_time = current_time

    time.sleep(0.1)  # Dừng một chút để CPU không bị quá tải
