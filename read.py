from restructure import *

if __name__ == "__main__":
    frame = cv2.imread("box_test_images\\0.jpg")
    frame = get_box(frame)
    string = find_digits_str(frame)
    for i in range(len(string)):
        cv2.namedWindow("string", 0)
        cv2.imshow("string", string[i])
        cv2.waitKey(0)
        digits = split_digits_str(string[i])
        print(len(digits))
        for j in range(len(digits)):
            cv2.namedWindow("digit", 0)
            cv2.imshow("digit", digits[j])
            cv2.waitKey(500)
    print(len(string))