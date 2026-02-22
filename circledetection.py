import numpy as np
import cv2
import os

image_path = os.path.join('.','Images', 'dartboard1.jpg')
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (17, 17), 0)



def nothing(x):
    pass

cv2.namedWindow("Tuner")
cv2.createTrackbar("param1", "Tuner", 50, 200, nothing)
cv2.createTrackbar("param2", "Tuner", 37, 100, nothing)
cv2.createTrackbar("minR",   "Tuner", 50, 300, nothing)
cv2.createTrackbar("maxR",   "Tuner", 200, 500, nothing)

while True:
    p1   = cv2.getTrackbarPos("param1", "Tuner")
    p2   = cv2.getTrackbarPos("param2", "Tuner")
    minR = cv2.getTrackbarPos("minR",   "Tuner")
    maxR = cv2.getTrackbarPos("maxR",   "Tuner")

    display = img.copy()

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1,
                                minDist=100, param1=p1, param2=p2,
                                minRadius=minR, maxRadius=maxR)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(display, (x, y), r, (0, 255, 0), 3)
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Tuner", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
