import os
import cv2
import numpy as np

READ_FOLDER = 'dataset'  # 需要处理的图片，经过矫正的
WRITE_FOLDER = 'write_folder'  # 最后得到的一个个的数字的图片
PROCESS_FOLDER = 'process'  # 被二值化、腐蚀之后的图片（预处理图片）
DIGITS_STR_FOLDER = 'strings'  # 一串一串的数字
# 调整以下参数可以获得合适的预处理图片
THRESHOLD = 120  # discriminate digits from background
CROP_LEN = 10  # get rid of original image's black borders
RESOLUTION = 28  # 28*28 for mnist dataset
kernel_connect = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
# Elliptical Kernel
kernel_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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


# 输入一串数字（图片）将其分割为单个的数字，注意单个数字的图片大小还不是28*28
def split_digits_str(s, prefix_name, is_vertical):
    # to read digits of a string in order, rotate the image
    # and let the leading digit lying in the bottom
    # since cv2.findContours from bottom to top
    if is_vertical:
        s = np.rot90(s, 2)
    else:
        s = np.rot90(s)

    # if each digit is continuous (not broken), needn't dilate
    # so for image 5/6/7, use (*); otherwise, use (**)
    # for image 0/1, iter=2; for image 2/3/4, iter=1
    s_copy = cv2.dilate(s, kernel_connect, iterations=1)  # (**)

    s_copy2 = s_copy.copy()
    im_s, contours, hierarchy = cv2.findContours(s_copy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    idx = 0
    digits_arr = np.array([])
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
        digit_name = os.path.join(WRITE_FOLDER, prefix_name + str(idx) + '.png')
        cv2.imwrite(digit_name, digit)

        # a digit: transform 2D array to 1D array
        digit = np.concatenate([(digit[i]) for i in range(RESOLUTION)])
        digits_arr = np.append(digits_arr, digit)

    # transform 1D array to 2D array
    # digits_arr = digits_arr.reshape((digits_arr.shape[0] / (RESOLUTION * RESOLUTION), -1))
    # 上面的语句加上后报错，也不知道到底做什么用的


def load_digits_arr_from_folder():  # 输出整理好的单个数字的串
    digits_arr = np.array([])
    for filename in os.listdir(WRITE_FOLDER):
        img = cv2.imread(os.path.join(WRITE_FOLDER, filename), 0)
        fn = os.path.splitext(filename)[0]  # without extension
        if img is not None:
            digit = np.concatenate([(img[i]) for i in range(RESOLUTION)])
            digits_arr = np.append(digits_arr, digit)
    digits_arr = digits_arr.reshape((-1, RESOLUTION * RESOLUTION))
    return digits_arr


def find_digits_str(folder):  # 找出所有的数字串，这个对我们可能没有用处，因为我们总共只有一串
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        fn = os.path.splitext(filename)[0]  # without extension
        if img is not None:
            height, width = img.shape
            thre_name = os.path.join(PROCESS_FOLDER, fn + '_thre.png')
            cropped = img[CROP_LEN:height - CROP_LEN, CROP_LEN:width - CROP_LEN]
            ret, thresh_img = cv2.threshold(cropped, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            is_vertical = is_vertical_writing(thresh_img)
            dilated = cv2.dilate(thresh_img, kernel_ellip, iterations=1)
            if is_vertical:
                dilated = cv2.dilate(dilated, kernel_cross_h, iterations=10)
            else:
                dilated = cv2.dilate(dilated, kernel_cross_w, iterations=10)
            im_d, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            idx = 0
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                if is_vertical and (w < 30 or w > 100 or h < 70 or h > 520):
                    continue
                elif (is_vertical == False) and (h < 30 or h > 100 or w < 70 or w > 520):
                    continue

                idx = idx + 1
                digits_str = thresh_img[y:y + h, x:x + w]
                save = np.rot90(digits_str)
                str_name = os.path.join(DIGITS_STR_FOLDER, fn + str(idx) + "str.png")
                cv2.imwrite(str_name, save)
                # digits_arr = split_digits_str(digits_str, fn + '_s' + str(idx), is_vertical)
                cv2.rectangle(thresh_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.imwrite(thre_name, thresh_img)


find_digits_str(READ_FOLDER)
for filename in os.listdir(DIGITS_STR_FOLDER):
        digits_string = cv2.imread(os.path.join(DIGITS_STR_FOLDER, filename), 0)
        split_digits_str(digits_string, filename, is_vertical_writing(digits_string))

