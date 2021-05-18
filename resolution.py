"""
1. Code that takes in counts @ native resolution
2. Upscales that raw frame x2
3. Runs that through scale/crop 
4. Upscales that
5. Exports 
"""


import mediapipe as mp
import numpy as np
import cv2
import os, random
from utils import draw_hands, bbox, get_centroid
import PIL
from PIL import Image

import cv2
from cv2 import dnn_superres
import numpy as np
import os
from moviepy.editor import *
from videowriter import frame_cap, convert_frames_to_video

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
amount_scale = 2
model = "FSRCNN_x"
model_src = "fsrcnn"
path = model + str(amount_scale) + ".pb"
sr.readModel(path)

#For this test, only using ones that autozoom/scale 

#count1, count2, count5, count7, count13

base = "../ThesisData/"

files = ['1_overlap/1_count1_overlap.mp4', '2_overlap/count5_overlap.mov', 
'3_overlap/3_count8_overlap.mov', '4_overlap/4_count7_overlap.mov', 
'5_overlap/5_count1_overlap.mp4', '6_overlap/6_count2_overlap.mov',
'7_overlap/7_count13_overlap.mov', '8_overlap/8_count16_overlap.mov', 
'9_overlap/9_count2_overlap.mp4', '10_overlap/10_count8_overlap.mov']

"""
Goes through each file and upscales it and saves upscale to be 
at least HD resolution -  1920 X 1080 based off native resolution or not
"""
def upscale_originals(files):
	# Set the desired model and scale to get correct pre- and post-processing
	sr.setModel("edsr", 3)

	# Upscale the image
	result = sr.upsample(image)

	# Save the image
	cv2.imwrite("./upscaled.png", result)
	return ""

"""
Centers and exports to 4:3 sizing
"""
def zoom_and_crop(file, zoom_ratio=0):
	return ""


# loading the image
img = PIL.Image.open("geeksforgeeks.png")
  
# fetching the dimensions
wid, hgt = img.size
  
# displaying the dimensions
print(str(wid) + "x" + str(hgt))