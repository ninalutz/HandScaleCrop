import cv2
from cv2 import dnn_superres

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('rawframes/1/0.jpg')

# Read the desired model
path = "FSRCNN_x2.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("fsrcnn", 2)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite("upscaled.jpg", result)