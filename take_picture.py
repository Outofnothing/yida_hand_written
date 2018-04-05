import cv2
cap = cv2.VideoCapture(1)
idx = 0
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
while True:
    ret, frame = cap.read()
    cv2.namedWindow("frame", 0)
    cv2.imshow("frame", frame)
    print(frame.shape)
    cv2.waitKey(10)

    """if cv2.waitKey(20) == 27:
        cv2.imwrite("box_test_images\\%s.jpg" % idx, frame)
        idx += 1"""