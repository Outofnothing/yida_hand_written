import cv2
import time
cap = cv2.VideoCapture(1)
idx = 0
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
cap.set(3,1280) 
cap.set(4,720)  
time.sleep(2)
while True:
    ret, frame = cap.read()
    cv2.namedWindow("frame", 0)
    cv2.imshow("frame", frame)
    print(frame.shape)
    if cv2.waitKey(50) == 27:
        cv2.imwrite("box_test_images\\0409.jpg", frame)
        break
cv2.destroyAllWindows()