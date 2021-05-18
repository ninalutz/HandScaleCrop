"""
1. Code that takes in counts @ native resolution
2. Upscales that raw frame x2
3. Runs that through scale/crop 
4. Upscales that
5. Exports 
"""


import mediapipe as mp
import numpy as np
import os, random
from utils import draw_hands, bbox, get_centroid

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

mp_drawing.draw_landmarks

import cv2
from cv2 import dnn_superres
import numpy as np
import os
from moviepy.editor import *
from videowriter import frame_cap, convert_frames_to_video

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()
path = "FSRCNN_x2.pb"
sr.readModel(path)
sr.setModel("fsrcnn", 2)


#For this test, only using ones that autozoom/scale 

#count1, count2, count5, count7, count13

base = "../ThesisData/"

files = ['1_overlap/1_count1_overlap.mp4', '2_overlap/count5_overlap.mov', 
'3_overlap/3_count8_overlap.mov', '4_overlap/4_count7_overlap.mov', 
'5_overlap/5_count1_overlap.mp4', '6_overlap/6_count2_overlap.mov',
'7_overlap/7_count13_overlap.mov', '8_overlap/8_count16_overlap.mov', 
'9_overlap/9_count2_overlap.mp4', '10_overlap/10_count8_overlap.mov']

zooms = [5.897253787878788, 2.943646771771774, 9,  
5.175111454046642,5.7, 4, 4, 4.1, 4, 10]



"""
Goes through list of frames and upscales it and saves upscale
"""
def upscale_frames(pathIn, index):
	files = [f for f in os.listdir(pathIn)]
	for f in files:
		if f[-1] != "g":
			files.remove(f)
	for i in range(len(files)):
		filename=pathIn + str(i) + ".jpg"
		#reading each files
		img = cv2.imread(filename)
		result = sr.upsample(img)
		cv2.imwrite("upscales/" + str(index) +"/" + str(i) + ".jpg", result)
	print("Done upscaling: " + str(index))
	convert_frames_to_video("upscales/" + str(index) +"/", "upscale_vids/" + str(index) +".mov")


"""
Centers and exports to 4:3 sizing
"""
def zoom_and_crop(file, zoom_ratio, index):

	aspect_ratio = '4_3_'
	write_out_width = 1024*2
	write_out_height = 768*2

	center_assigned = False

	count = 0

	# Initiate holistic model
	with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
		
		while cap.isOpened():
			ret, frame = cap.read()
			
			try:
				h, w, c = frame.shape
			except:
				cap.release()
				# result.release()
				break
				print("HELLO")

			# Recolor Feed
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# Make Detections
			results = holistic.process(image)

			# Recolor image back to BGR for rendering
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			draw_hands(image, results, mp_drawing, mp_holistic)

			global_x_min = w
			global_y_min = h
			global_x_max = 0
			global_y_max = 0

			if results.left_hand_landmarks or results.right_hand_landmarks:
				if results.left_hand_landmarks:
					x_max = 0
					y_max = 0
					x_min = w
					y_min = h
					for data_point in results.left_hand_landmarks.landmark:
						x, y = int(data_point.x * w), int(data_point.y * h)
						if x > global_x_max:
							global_x_max = x
						if x < global_x_min:
							global_x_min = x
						if y > global_y_max:
							global_y_max = y
						if y < global_y_min:
							global_y_min = y
				if results.right_hand_landmarks:
					x_max = 0
					y_max = 0
					x_min = w
					y_min = h
					for data_point in results.right_hand_landmarks.landmark:
						x, y = int(data_point.x * w), int(data_point.y * h)
						if x > global_x_max:
							global_x_max = x
						if x < global_x_min:
							global_x_min = x
						if y > global_y_max:
							global_y_max = y
						if y < global_y_min:
							global_y_min = y

			box = bbox(global_x_min, global_x_max, global_y_min, global_y_max)

			if not center_assigned and global_x_max - global_x_min > 0:
				# print("HELLO")
				center = get_centroid(global_x_min, global_x_max, global_y_min, global_y_max)
				center_assigned = True

			cv2.circle(image, (center[0],center[1]), radius=2, color=(0, 0, 255), thickness=2)

			cv2.imshow('Video', image)

			x1 = center[0] - write_out_width/zoom_ratio
		
			if x1 < 0:
				x1 = 0

			x2 = center[0] + write_out_width/zoom_ratio
		
			if x2 > write_out_width:
				x2 = write_out_width
			
			y1 = center[1] - write_out_height/zoom_ratio
		
			if y1 < 0:
				y1 = 0
		
			y2 = center[1] + write_out_height/zoom_ratio
			
			if y2 > write_out_height:
				y2 = write_out_height
		
			pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
			pts2 = np.float32([[0,0],[write_out_width,0],[0,write_out_height],[write_out_width,write_out_height]])

			M = cv2.getPerspectiveTransform(pts1,pts2)

			dst = cv2.warpPerspective(frame,M,(write_out_width,write_out_height))

			# if ret == True  and write_out == True: 
			# 	result.write(dst)
					
			cv2.imshow('Zoomed',dst)

			#TODO - write out these frames! 
			cv2.imwrite("upscale_crops/" + str(index) + "/%d.jpg" % count, dst)

			count += 1

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

	cap.release()
	# result.release()
	cv2.destroyAllWindows()


current_number = 10

test_file = 'upscale_vids/' + str(current_number) + ".mov"

cap = cv2.VideoCapture(test_file)

# for i in range(1,11):
test_file = 'upscale_vids/' + str(current_number) + ".mov"
cap = cv2.VideoCapture(test_file)
zoom_and_crop(test_file, zooms[current_number-1], current_number)
convert_frames_to_video('upscale_crops/' + str(current_number) + "/", 'upscale_results/' + str(current_number) + ".mov")
current_number = i


#Crop frames and save cropped frames
#Upscale result of these frames 
#Stitch those upscales into video