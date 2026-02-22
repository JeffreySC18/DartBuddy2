import cv2
import numpy as np
import os

image_path = os.path.join('.','Images', 'dartboard1.jpg')
img = cv2.imread(image_path)

src_points = np.float32([
    [222, 87],   # top-left of board
    [390, 86],   # top-right
    [398, 301],  # bottom-right
    [226, 308]   # bottom-left
])

# Where we want those points to map to
output_size = 500
dst_points = np.float32([
    [0, 0],
    [output_size, 0],
    [output_size, output_size],
    [0, output_size]
])

# Compute the transformation matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply it
warped = cv2.warpPerspective(img, M, (output_size, output_size))

cv2.imshow("Original", img)
cv2.imshow("Corrected", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()