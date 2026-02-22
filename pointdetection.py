import cv2
import numpy as np
import os

image_path = os.path.join('.','Images', 'dartboard1.jpg')
img = cv2.imread(image_path)

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        cv2.imshow("Select Corners", img)
        print(f"Point {len(points)}: ({x}, {y})")

cv2.imshow("Select Corners", img)
cv2.setMouseCallback("Select Corners", click)

print("Click the 4 corners of the dartboard (top-left, top-right, bottom-right, bottom-left)")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Your points:", points)