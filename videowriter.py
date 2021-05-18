import numpy as np
import cv2
import os
from moviepy.editor import *

fourcc = cv2.VideoWriter_fourcc(*'X264')
fps = 30 #constant 30 fps 

base = "Variations_5_14_21/"
gaus_paths = ["gaus_3", "gaus_5", "gaus_9", "gaus_13", "gaus_21"]
med_paths =  ["med_3", "med_5", "med_7", "med_9", "med_11", "med_13", "med_15", "med_21"]
sharp_paths = []

values = [3, 5, 7, 9]


# values = {"gaus_3":3, "gaus_5":5, "gaus_9":9, "gaus_13":13, "gaus_21":21, "med_3":3, "med_5":5, "med_7":7, "med_9":9, 
# "med_11":11, "med_13":13,"med_15":15, "med_21":21}

"""
Divides a video at a pre-formatted path 
into raw frames as .jpg  
"""
def frame_cap(path, index):
	cap = cv2.VideoCapture(path)

	count = 0
	while(True):
	    ret, frame = cap.read()
	    success, image = cap.read()

	    cv2.imwrite("rawframes/" + str(index) + "/%d.jpg" % count, image)
	    count += 1
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

"""
Takes in a folder of raw frames
Sttiches them into a video file
"""
def convert_frames_to_video(pathIn,pathOut):
    frame_array = []
    files = [f for f in os.listdir(pathIn)]
    files = sorted(files)
    for f in files:
        if f[-1] != "g":
            files.remove(f)
    for i in range(len(files)):
        filename=pathIn + str(i) + ".jpg"
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,fourcc, fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


"""
Stitches video files in paths into one video
"""
def stitch_videos(paths, result_name):
    clips = []
    for path in paths:
        clip = VideoFileClip(path)
        clips.append(clip)
        # concatinating both the clips
    
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(result_name+".mp4")

# stitch_videos(sharp_paths, "sharpened")


for val in values:
    paths = []
    result = ""
    for i in range(1,11):
        paths.append("Variations_5_14_21/sharpen_median_" + str(val) + "/" + str(i) + ".mov")
        result_name = "sharpen_median_" + str(val) + "_count.mov" 
    stitch_videos(paths, result_name)
    paths = []
    for i in range(1, 11):
        paths.append("Variations_5_14_21/sharpen_gaus_" + str(val) + "/" + str(i) + ".mov")
        result_name = "sharpen_gaus_" + str(val) + "_count.mov" 
    stitch_videos(paths, result_name)