import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_canny_edge_detection(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # 使用Otsu方法自动计算阈值
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 应用Canny边缘检测
    edges = cv2.Canny(thresh_image, 50, 150)

    # 显示原始图像和边缘检测结果
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges detected by Otsu-Canny')
    plt.axis('off')

    plt.show()

# 调用函数，输入图像路径
otsu_canny_edge_detection('./test.jpg')

