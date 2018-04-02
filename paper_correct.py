import cv2
import numpy as np
from function import adaptive_Thresh

filename = 'box_test_images\\test.jpg'
CROP_LEN = 400  # 被去掉的A4纸边缘的长度， 注意是每边
kernel_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
cropped = gray[CROP_LEN:height - CROP_LEN, CROP_LEN:width - CROP_LEN]
# 采用自适应阈值法，显著减少图片不连续或糊成一团的现象
thresh_img = adaptive_Thresh(cropped, (39, 39))
dilated = cv2.dilate(thresh_img, kernel_ellip, iterations=2)
im_d, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_biggest = None
length = 0
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    if (w + h) > length:
        contour_biggest = contour
        length = w + h
cv2.drawContours(img, contour_biggest, -1, (0, 255, 255), 4)
[x, y, w, h] = cv2.boundingRect(contour_biggest)
output = thresh_img[y:y+h, x:x+w]
assert dilated.shape == thresh_img.shape
cv2.namedWindow('outcome', 0)
cv2.imshow("outcome", img)
cv2.waitKey(0)
cv2.imshow("outcome", thresh_img)
cv2.waitKey(0)
cv2.imshow("outcome", output)
cv2.waitKey(0)
cv2.imshow("outcome", dilated)
cv2.waitKey(0)
