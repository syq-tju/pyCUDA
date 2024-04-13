# Compare this snippet from pyCUDA/Chapter3_CV/3_2_CV_canny.py:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('./test.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found.")
    exit()

# 使用高斯滤波平滑图像
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 显示原始图像和边缘检测结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edges detected by Canny')
plt.axis('off')

plt.show()
