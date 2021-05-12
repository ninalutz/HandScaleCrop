import mediapipe as mp
import numpy as np
import cv2
from utils import draw_hands, bbox, get_centroid

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

mp_drawing.draw_landmarks
zoom_count = 0
# cap = cv2.VideoCapture('3_overlap/3_count2_overlap.mov')
# cap = cv2.VideoCapture('1_overlap/1_count1_overlap.mp4')
# cap = cv2.VideoCapture('2_overlap/count4_overlap.mov') #this one isn't working right
# cap = cv2.VideoCapture('2_overlap/count5_overlap.mov')
# cap = cv2.VideoCapture('9_overlap/9_count6_overlap.mov') #this one is also weird
# cap = cv2.VideoCapture('4_overlap/4_count7_overlap.mov') 
cap = cv2.VideoCapture('../ThesisData/4_overlap/4_count8_overlap.mov') #not working AT ALL so need to look into this method



aspect_ratio = '16_9_'
write_out_width =1280
write_out_height = 720

top_margin = 0.08

# aspect_ratio = '16_10_'
# write_out_width =1680
# write_out_height = 1050

# aspect_ratio = '4_3_'
# write_out_width = 1024
# write_out_height = 768
zoom_ratio = 1

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



def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('x = %d, y = %d'%(x, y))


# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        try:
            h, w, c = frame.shape
        except:
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

        #TODO - more sophisticated way to get centroid for whole clip 
        if not center_assigned and global_x_max - global_x_min > 0:
            # print("HELLO")
            center = get_centroid(global_x_min, global_x_max, global_y_min, global_y_max)
            center_assigned = True

        cv2.circle(image, (center[0],center[1]), radius=2, color=(0, 0, 255), thickness=2)

        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), (0, 255, 0), 2)

        print("Distance from bounding box top to 0: " + str(box[2]) + " percent: " + str(box[2]/h))

        zoom_ratio = (box[2]/h)/top_margin
        print(zoom_ratio)

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

        # cv2.line(dst, (0,int(top_margin*h)), (w,int(top_margin*h)), color=(0, 0, 255), thickness=2)

        cv2.line(dst, (0,int(box[2]*zoom_ratio)), (w,int(box[2]*zoom_ratio)), color=(255, 0, 255), thickness=2)

        if ret == True: 
            result.write(dst)
                
        cv2.imshow('Zoomed',dst)

        cv2.setMouseCallback('Video',draw_circle)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
result.release()
cv2.destroyAllWindows()

