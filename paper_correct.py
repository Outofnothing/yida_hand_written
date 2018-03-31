import cv2
import numpy as np
from function import adaptive_Thresh

filename = 'box_test_images\\test.jpg'
CROP_LEN = 100  # 被去掉的A4纸边缘的长度， 注意是每边
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img.shape
cropped = img[CROP_LEN:height - CROP_LEN, CROP_LEN:width - CROP_LEN]
# 采用自适应阈值法，显著减少图片不连续或糊成一团的现象
thresh_img = adaptive_Thresh(cropped, (31, 31))
cv2.namedWindow('outcome', 0)
cv2.imshow("outcome", thresh_img)
cv2.waitKey(0)