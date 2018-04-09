import cv2
import numpy as np
from restructure import *

# 调整以下参数可以获得合适的预处理图片

RESOLUTION = 28  # 28*28 for mnist dataset
kernel_connect = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
def _split_digits_str(s):
    # 输入的数字串从下往上列出，右视图是阅读方向
    s_copy = cv2.dilate(s, kernel_connect, iterations=1)
    string_h, string_w = s_copy.shape
    im_s, contours, hierarchy = cv2.findContours(s_copy.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    idx = 0
    location = 1000
    digits_arr = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if x > int(0.75 * string_w):
            location = idx
        idx = idx + 1
        digit = s_copy[y:y + h, x:x + w]
        # in order to keep the original scale of digit
        # pad rectangles to squares before resizing
        pad_len = int((h - w) / 2)
        if pad_len > 0:  # to pad width
            # Forms a border around an image: top, bottom, left, right
            digit = cv2.copyMakeBorder(digit, 0, 0, pad_len, pad_len, cv2.BORDER_CONSTANT, value=0)
        elif pad_len < 0:  # to pad height
            digit = cv2.copyMakeBorder(digit, -pad_len, -pad_len, 0, 0, cv2.BORDER_CONSTANT, value=0)
        pad = int(digit.shape[0] / 5)  # avoid the digit directly connect with border, leave around 4 pixels
        digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.resize(digit, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        digit = np.rot90(digit, 3)  # rotate back to horizontal orientation
        digits_arr.append(digit)
    return digits_arr, location


frame = cv2.imread("box_test_images\\0.jpg")
frame = get_box(frame)
string = find_digits_str(frame)
for i in range(len(string)):
    cv2.namedWindow("string", 0)
    cv2.imshow("string", string[i])
    cv2.waitKey(0)
    digits, location = is_point.split_digits_str(string[i])
    print(len(digits))
    for j in range(len(digits)):
        cv2.namedWindow("digit", 0)
        cv2.imshow("digit", digits[j])
        cv2.waitKey(500)