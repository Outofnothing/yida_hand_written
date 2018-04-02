from restructure import *

if __name__ == "__main__":
    frame = cv2.imread("box_test_images\\test.jpg")
    frame = get_box(frame)
    frame = find_digits_str(frame)
    for i in range(len(frame)):
        cv2.imshow("string", frame[i])
        cv2.waitKey(0)
