"""
Draws landmarks and segments
"""
def draw_hands(image, results, mp_drawing, mp_holistic):   
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )


"""
Returns coordinates for bounding box
TODO: Fix min measurements to center better
TODO: Fix measurements for closed hand/hand by number 
"""
def bbox(global_x_min, global_x_max, global_y_min, global_y_max):
    if global_x_max - global_x_min > global_y_max - global_y_min:
        delta = (global_x_max - global_x_min) - (global_y_max - global_y_min)
        per_increase =  int(0.5*delta)
        global_y_max += int(delta/2)
        global_y_min -= int(delta/2)

        global_x_max += per_increase
        global_y_max += per_increase
        global_x_min -= per_increase
        global_y_min -= per_increase

        delta = (global_x_max - global_x_min) - (global_y_max - global_y_min)

        global_y_max += delta

        #print(global_x_max - global_x_min)
    else:
        delta = (global_y_max - global_y_min) - (global_x_max - global_x_min)
        per_increase =  int(0.5*delta)
        global_x_max += int(delta/2)
        global_x_min -= int(delta/2)

        global_x_max += per_increase
        global_y_max += per_increase
        global_x_min -= per_increase
        global_y_min -= per_increase

        delta = (global_y_max - global_y_min) - (global_x_max - global_x_min)

        global_x_max += delta

       #print(global_x_max - global_x_min)
    return [max(1,global_x_min), max(1,global_x_max), max(1,global_y_min), max(1,global_y_max)]


"""
Returns the centerpoint of BBOX
"""
def get_centroid(global_x_min, global_x_max, global_y_min, global_y_max):
    return [int((global_x_min+global_x_max)/2), int((global_y_min+global_y_max)/2)]

"""
Writes out the cropped video to file
TODO: Finish implementing
"""
def write_out(filename, cap):
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    result = cv2.VideoWriter('filename.mp4', fourcc, fps, size)

    while(True):
        ret, frame = video.read()
      
        if ret == True: 
            # Write the frame into the
            # file 'filename.avi'
            result.write(frame)
      
            # Display the frame
            # saved in the file
            cv2.imshow('Frame', frame)
      
            # Press S on keyboard 
            # to stop the process
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
      
        # Break the loop
        else:
            break

def plot_bbox(delta):
    return "" 