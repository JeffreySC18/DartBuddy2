import cv2
import os

'''#read image
image_path = os.path.join('.','Images', 'dartboard1.jpg')
img = cv2.imread(image_path)

#write image
write_path = os.path.join('.','Images', 'dartboard1_copy.jpg')
cv2.imwrite(write_path, img)

#Visualize
cv2.imshow('image,', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)
print(img.dtype)'''

'''cap = cv2.VideoCapture(0)
ret = True
while ret:
    

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Webcam Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()'''

'''# color spaces
image_path = os.path.join('.','Images', 'dartboard1.jpg')
img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("HSV", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Subtract the previous frame from the current one
    diff = cv2.absdiff(prev_gray, gray)

    # Threshold — anything with a difference > 25 becomes white, rest black
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    cv2.imshow("Motion", thresh)
    cv2.imshow("Live", frame)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


