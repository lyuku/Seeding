import cv2
import numpy as np
import matplotlib.pyplot as plt


# 提取照片中HSV颜色空间绿色部分
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# 用提取的绿色mask处理图片，过滤背景
def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


# 提高图片锐度
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


img = cv2.imread('./input/plant-seedlings-classification/test/0c4199daa.png', cv2.IMREAD_COLOR)

img = cv2.resize(img, (249, 249), interpolation=cv2.INTER_CUBIC)

img1 = segment_plant(img)

img2 = sharpen_image(img1)

print(img.shape)
print(img2.shape)

fig, axs = plt.subplots(1, 3, figsize=(20, 20))

axs[0].imshow(img)
axs[1].imshow(img1)
axs[2].imshow(img2)

plt.show()