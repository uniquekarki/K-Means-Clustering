import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('Input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixel_vals = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)

k = 7
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((image.shape))
segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

cv2.imwrite('Result.jpg', segmented_image)

print('Image saved successfully! Result image saved as Result.jpg')
a = input('Press any key to exit')
if a:
    exit(0)
