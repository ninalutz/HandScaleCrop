import mediapipe as mp
import numpy as np
import cv2

def zoom_frame_old(frame, box, center, w, h, top_margin, frames, frameCount, zooms, write_out_width, write_out_height): 
    zoom_ratio = (box[2]/h)/top_margin
    
    frames.append(frameCount)
    zooms.append(zoom_ratio)
    frameCount += 1

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

    #TODO - figure out write out
    # if ret == True  and write_out == True: 
        # result.write(dst)
    
    return dst