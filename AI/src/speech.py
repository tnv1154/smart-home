import pygame
import os
import time

Base_path = "E:/PythonProjectMain/"
Voice_path = os.path.join(Base_path,"AI/Voice/")

class Speech:
    def __init__(self):
        pygame.mixer.init()
        self.nhin_thang = pygame.mixer.Sound(os.path.join(Voice_path, "Vui_long_nhin_thang.wav"))
        self.xoay_phai = pygame.mixer.Sound(os.path.join(Voice_path, "Vui_long_xoay_phai.wav"))
        self.xoay_trai = pygame.mixer.Sound(os.path.join(Voice_path, "Vui_long_xoay_trai.wav"))
        self.trong_khung = pygame.mixer.Sound(os.path.join(Voice_path, "Vui_long_dat_khuon_mat_trong_khung.wav"))

    def Nhin_thang_start(self):
        self.nhin_thang.play()
        time.sleep(0.001)

    def Xoay_phai_start(self):
        self.xoay_phai.play()
        time.sleep(0.0001)

    def Xoay_trai_start(self):
        self.xoay_trai.play()
        time.sleep(0.0001)


    def Trong_khung_start(self):
        self.trong_khung.play()
        time.sleep(0.0001)


def main():
    speech = Speech()
    last_play_time = time.time()
    for i in range(10000000):
        print(i)
        current_time = time.time()
        if current_time - last_play_time >= 5:
            speech.Trong_khung_start()
            last_play_time = current_time
        #speech.Nhin_thang_start()

if __name__ == "__main__":
    main()
