
import mediapipe as mp
import numpy as np
import cv2
from utils import draw_hands, bbox, get_centroid, solve_globals, draw_debugs

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


import plotly.express as px


mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

mp_drawing.draw_landmarks
zoom_count = 0
# test_file = '3_overlap/3_count1_overlap.mp4'

test_file = '3_overlap/3_count2_overlap.mov'
# test_file = '2_overlap/count5_overlap.mov'

# test_file = '4_overlap/4_count7_overlap.mov'
# test_file = '5_overlap/5_count13_overlap.mov'


#Strug buses
# test_file = '2_overlap/count4_overlap.mov'
# test_file = '9_overlap/9_count6_overlap.mov' #this one is also weird
# test_file = '4_overlap/4_count8_overlap.mov' #not working AT ALL so need to look into this method
# test_file = '7_overlap/7_count16_overlap.mov'

# cap = cv2.VideoCapture('../ThesisData/)
cap = cv2.VideoCapture('../ThesisData/' + test_file)

# cap = cv2.VideoCapture('../ThesisData/4_overlap/4_count8_overlap.mov') #not working AT ALL so need to look into this method


zooms = []
frames = []
frameCount = 0

aspect_ratio = '16_9_'
write_out_width =1280
write_out_height = 720

#determines if clip is written out or not
write_out = False

#Percentage of frame for top margin
top_margin = 0.08

# aspect_ratio = '16_10_'
# write_out_width =1680
# write_out_height = 1050

# aspect_ratio = '4_3_'
# write_out_width = 1024
# write_out_height = 768
zoom_ratio = 1
zoom_assigned = False

# aspect_ratio = '3_2_'
# write_out_width = 1080
# write_out_height = 720

center_assigned = False
center = [620, 400]

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
size = (write_out_width, write_out_height)
fourcc = cv2.VideoWriter_fourcc(*'X264')
fps = cap.get(cv2.CAP_PROP_FPS)
result = cv2.VideoWriter(aspect_ratio+'test.mp4', fourcc, fps, size)

debug_draw = True

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        try:
            h, w, c = frame.shape
        except:
            cap.release()
            result.release()
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

        cur_globals = [w, h, 0, 0]

        if results.left_hand_landmarks or results.right_hand_landmarks:
            if results.left_hand_landmarks:
                cur_globals = solve_globals(results.left_hand_landmarks.landmark, "left", global_x_min, global_y_min, global_x_max, global_y_max, w, h)
            if results.right_hand_landmarks:
                cur_globals = solve_globals(results.right_hand_landmarks.landmark, "right", global_x_min, global_y_min, global_x_max, global_y_max, w, h)
            global_x_min = cur_globals[0]
            global_y_min = cur_globals[1]
            global_x_max = cur_globals[2]
            global_y_max = cur_globals[3]

        box = bbox(global_x_min, global_x_max, global_y_min, global_y_max)

        if not center_assigned and global_x_max - global_x_min > 0:
            center = get_centroid(global_x_min, global_x_max, global_y_min, global_y_max)
            center_assigned = True


        if debug_draw:
            draw_debugs(image, center, box, w, h, top_margin)

        print("Distance from bounding box top to 0: " + str(box[2]) + " percent: " + str(box[2]/h))



        zoom_ratio = (box[2]/h)/top_margin
        
        frames.append(frameCount)
        zooms.append(zoom_ratio)
        frameCount += 1

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

        #line for the top margin
        cv2.line(dst, (0,int(top_margin*h)), (w,int(top_margin*h)), color=(0, 0, 255), thickness=2)

        if ret == True  and write_out == True: 
            result.write(dst)
                
        cv2.imshow('Zoomed',dst)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
result.release()
cv2.destroyAllWindows()

average_zoom = sum(zooms)/len(zooms)

print("Average zoom: " + str(average_zoom))


fig = px.scatter(x=frames, y=zooms,  title=test_file.split('/')[1] + " | " + "Average zoom: " + str(sum(zooms)/len(zooms)))

fig.show()
