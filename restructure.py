import cv2
import numpy as np


# 调整以下参数可以获得合适的预处理图片

RESOLUTION = 28  # 28*28 for mnist dataset
kernel_connect = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
# Elliptical Kernel
kernel_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
# Cross-shaped Kernel
# to manipulate the orientation of dilution, large x means
# horizonatally dilating more, large y means vertically dilating more
kernel_cross_h = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 8))
kernel_cross_w = cv2.getStructuringElement(cv2.MORPH_CROSS, (8, 1))


def is_vertical_writing(img):  # 判断图片里数字的书写方向，因为cv2.findContours是由下往上找的
    h, w = img.shape
    h_bin = np.zeros(h, np.uint16)  # 0 to 65535
    w_bin = np.zeros(w, np.uint16)
    x, y = np.where(img == 255)  # white
    for i in x:
        h_bin[i] = h_bin[i] + 1
    for j in y:
        w_bin[j] = w_bin[j] + 1
    # calculate the number of continuous zero (background)
    # areas in vertical (h) and horizontal (w) orientation
    n_h_zero_area = 0
    for i in range(h - 1):
        if h_bin[i] == 0 and h_bin[i + 1] != 0:
            n_h_zero_area = n_h_zero_area + 1
    n_w_zero_area = 0
    for i in range(w - 1):
        if w_bin[i] == 0 and w_bin[i + 1] != 0:
            n_w_zero_area = n_w_zero_area + 1

    if n_h_zero_area > n_w_zero_area:  # sparse vertically
        return True  # dense horizontally
    return False


# 输入一串数字（图片）将其分割为单个的数字
def split_digits_str(s):
    # 输入的数字串从下往上列出，右视图是阅读方向
    s_copy = cv2.dilate(s, kernel_connect, iterations=1)
    im_s, contours, hierarchy = cv2.findContours(s_copy.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    idx = 0
    digits_arr = []
    for contour in contours:
        idx = idx + 1
        [x, y, w, h] = cv2.boundingRect(contour)
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
    return digits_arr


def find_digits_str(img):  # 找出所有的数字串，这个对我们可能没有用处，因为我们总共只有一串
    if img is not None:
        CROP_LEN = 20  # 被去掉的img 的边缘长度， 消掉边框
        height, width = img.shape
        cropped = img[CROP_LEN:height - CROP_LEN, CROP_LEN:width - CROP_LEN]  # 裁去边缘
        # eroded = cv2.erode(cropped, kernel_erode, iterations=2)
        dilated = cv2.dilate(cropped, kernel_ellip, iterations=1)
        # 输入的数字书写方向是水平的， 所以膨胀也要按照宽度方向膨胀
        dilated = cv2.dilate(dilated, kernel_cross_w, iterations=10)
        im_d, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        output = []
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            if h < 10 or w < 10:
                continue
            # 这里是用来判断contour大小的，以去掉明显不属于数字串的contour，需要根据环境调节
            digits_str = cropped[y:y + h, x:x + w]
            save = np.rot90(digits_str)
            output.append(save)
        output.reverse()
        return output


def adaptive_thresh(image, win_size, ratio=0.15):  # 定义了自适应阈值方法
    i_mean = cv2.boxFilter(image, cv2.CV_32FC1, win_size)
    out = image - (1.0 - ratio) * i_mean
    out[out >= 0] = 0  # 注意mnist数据集以黑色为底色
    out[out < 0] = 255
    out = out.astype(np.uint8)
    return out


def get_box(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        crop_len_this = 20
        cropped = gray[crop_len_this:height - crop_len_this, crop_len_this:width - crop_len_this]
        # 采用自适应阈值法，显著减少图片不连续或糊成一团的现象
        thresh_img = adaptive_thresh(cropped, (31, 31))
        dilated = cv2.dilate(thresh_img, kernel_ellip, iterations=2)
        im_d, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_biggest = None
        length = 0
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            if (w + h) > length:
                contour_biggest = contour
                length = w + h
        # cv2.drawContours(img, contour_biggest, -1, (0, 255, 255), 4)
        [x, y, w, h] = cv2.boundingRect(contour_biggest)
        output = thresh_img[y:y+h, x:x+w]
        assert dilated.shape == thresh_img.shape
        cv2.namedWindow('outcome', 0)
        cv2.imshow("outcome", output)
        cv2.waitKey(0)
        return output

