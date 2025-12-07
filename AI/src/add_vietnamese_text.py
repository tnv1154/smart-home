import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class AddVietnameseText:
    def add_vietnamese_text(image, text, position, font_size = 24, font_color=(0, 255, 0)):
        """"Thêm text tiếng việt vào ảnh"""
        #chuyển bgr sang rgb
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text(position, text, font=font, fill=font_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    """test"""
    image = cv2.imread("E:/PythonProjectMain/AI/DataSet/FaceData/raw/1/1_001.png")
    image = AddVietnameseText.add_vietnamese_text(image, "Tiếng Việt", (10, 10))
    cv2.imshow("image", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()