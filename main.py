import mediapipe as mp
import numpy as np
import cv2
import os, random

from utils import draw_hands, bbox, get_centroid

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

mp_drawing.draw_landmarks
center_assigned = False

#Centers -- pre-assigned if not assigned in code
centers = [[0, 0], [0,0], [630, 370], [0, 0], [0,0],
[0, 0], [0,0], [0, 0], [600, 500], [630, 380], [0,0],
[620, 500],[0,0],[630, 440],[620, 400],[0, 0]]

#Temporary zoom ratios
zooms = {'4_3_':[7, 4.7, 6,6,4.5,4,5.5,
9,4,3.7,0,0,3.5,5,4,4.5], '3_2_':[6.2, 4.7, 6,6,4.5,4,5.5,
9,4,3.7,0,0,3.5,5,4,4.5], '16_10_':[9,6.3,8.7,8.5,4.7,5.7,
8,14,5.5,5,0,0,5.5,7,6,6.3], '16_9_':[6.5, 4.3, 6.2, 5.5, 
3.5, 4,6,9.5,4,3.6,0,0,3.5,4.7,4,4.2]}

write_out_widths = [1024, 1080, 1680, 1280]
write_out_heights = [768, 720, 1050, 720]

#Toggles whether the code should write results to file or not 
write_to_file = False

aspect_ratios = ['4_3_', '3_2_', '16_10_', '16_9_']

files = []
# for i in range(1, 11):
    #Gets a random file from the number file -- will change in final version 
    # file = random.choice(os.listdir("../ThesisData/" + str(i) + "_overlap/")) #change dir name to whatever
    # print("FILE: " + file)
    # print(file.split("count")[1].split("_")[0])
    # files.append(str(i) + "_overlap/" + file)

aspect_counter = 0

aspect_ratio = aspect_ratios[0]
write_out_width = write_out_widths[aspect_counter]
write_out_height = write_out_heights[aspect_counter]

file = "newtest.mp4"
# cap = cv2.VideoCapture(files[0])
cap = cv2.VideoCapture("archive.mp4")

current_number = 1
# count_number = int(file.split("count")[1].split("_")[0])
count_number = 1
zoom_ratio = zooms[aspect_ratio][count_number-1]
center = centers[count_number-1]


# print(files[0])
print(zoom_ratio)
print(count_number)
print(center)

if center[0] > 0:
    center_assigned = True

size = (write_out_width, write_out_height)
fourcc = cv2.VideoWriter_fourcc(*'X264')
fps = cap.get(cv2.CAP_PROP_FPS)
result = cv2.VideoWriter(aspect_ratio+ str(current_number) +'.mp4', fourcc, fps, size)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        try:
            h, w, c = frame.shape
        except:
            center_assigned = False

            current_number += 1

            if current_number == 11:
                current_number = 1
                aspect_counter += 1
                aspect_ratio = aspect_ratios[aspect_counter]
                write_out_width = write_out_widths[aspect_counter]
                write_out_height = write_out_heights[aspect_counter]

            file =  files[current_number-1]
            cap = cv2.VideoCapture(file)
            count_number = int(file.split("count")[1].split("_")[0])
            zoom_ratio = zooms[aspect_ratio][count_number-1]
            ret, frame = cap.read()
            h, w, c = frame.shape
            #reset writer params
            size = (write_out_width, write_out_height)
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            fps = cap.get(cv2.CAP_PROP_FPS)
            result = cv2.VideoWriter(aspect_ratio + str(current_number) +'_overlap.mp4', fourcc, fps, size)
            center = centers[count_number-1]

            if center[0] > 0:
                center_assigned = True

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

        #TODO - more sophisticated way to get centroid for whole clip 
        if not center_assigned and global_x_max - global_x_min > 0:
            center = get_centroid(global_x_min, global_x_max, global_y_min, global_y_max)
            center_assigned = True

        cv2.circle(image, (center[0],center[1]), radius=2, color=(0, 0, 255), thickness=2)

        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), (0, 255, 0), 2)

        #TODO -- make hands the same size by making zoom factor based off bounding box size

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


        if ret == True: 
            result.write(dst)
                
        cv2.imshow('Zoomed',dst)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if cv2.waitKey(10) & 0xFF == ord('s'):
            print("HELLO")
            cv2.imwrite(aspect_ratio + str(current_number) + ".png", dst)



cap.release()
result.release()
cv2.destroyAllWindows()
