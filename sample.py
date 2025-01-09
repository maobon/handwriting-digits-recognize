import os
import sys

import cv2 as cv
import imageio.v2 as imageio
import numpy as np
import tensorflow as tf
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

WINDOW_NAME = "Digits"


def predict_local_image(image_path):

    loaded_model = tf.keras.models.load_model("mnist_model_optimized_v9.keras")

    # 读取图片
    try:
        img_data = imageio.imread(image_path, pilmode="L")  # 加载图像并转换为灰度图
    except FileNotFoundError:
        print("Error: Image not found or could not be read.")
        return

    # 调整大小为28x28
    img_data = Image.fromarray(img_data).resize((28, 28))
    img_data = np.array(img_data)
    # 反转颜色（如果是白底黑字）
    img_data = 255 - img_data
    # 归一化处理
    img_data = img_data.astype("float32") / 255.0
    # 增加批次维度
    img_data = np.expand_dims(img_data, axis=0)
    # 预测
    prediction = loaded_model.predict(img_data)
    predicted_digit = np.argmax(prediction)
    print(f"== RESULT:Predicted digit for the image: {predicted_digit}")


def draw(event, x, y, flags, param):
    global img, pre_pts

    if event == cv.EVENT_LBUTTONDOWN:
        pre_pts = (x, y)

    if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        pts = (x, y)
        img = cv.line(img, pre_pts, pts, (0, 0, 0), thickness=10, lineType=cv.LINE_AA)
        pre_pts = pts
        cv.imshow(WINDOW_NAME, img)

    if event == cv.EVENT_LBUTTONUP:
        pre_pts = -1, -1

    if event == cv.EVENT_RBUTTONUP:
        print("okok...")
        if cv.imwrite("image.png", img):
            predict_local_image("image.png")


def restart():
    """重新初始化窗口"""
    global img, original_img
    img = original_img.copy()  # 重置为初始图像
    cv.imshow(WINDOW_NAME, img)  # 重新显示图像
    cv.setMouseCallback(WINDOW_NAME, draw)  # 设置鼠标回调


def main():
    global img, original_img

    # 加载图像
    img = cv.imread("bc_image.jpg")
    if img is None:
        print("Failed to load image")
        sys.exit()

    # 保存初始图像
    original_img = img.copy()

    # 显示图像窗口
    cv.imshow(WINDOW_NAME, img)
    cv.setMouseCallback(WINDOW_NAME, draw)

    while True:
        key = cv.waitKey(10) & 0xFF  # 监听键盘输入

        if key == ord("q"):
            cv.destroyAllWindows()
            break

        elif key == ord("r"):  # 按 'r' 键重置绘制
            print("Key 'r' detected! Resetting...")
            restart()  # 调用重置函数


if __name__ == "__main__":
    # 全局变量
    global img
    global original_img
    global pre_pts

    main()
