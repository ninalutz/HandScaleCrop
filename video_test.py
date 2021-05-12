import mediapipe as mp
import cv2
from utils import draw_hands, bbox, get_centroid

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

mp_drawing.draw_landmarks

cap = cv2.VideoCapture('../ThesisData/7_overlap/7_count10_overlap.mp4')

center_assigned = False
center = [0, 0]

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
	
	while cap.isOpened():
		ret, frame = cap.read()
		
		h, w, c = frame.shape

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

		#TODO - more sophisticated way to get centroid for whole clip 
		if not center_assigned and global_x_max - global_x_min > 0:
			center = get_centroid(global_x_min, global_x_max, global_y_min, global_y_max)
			# center_assigned = True

		box = bbox(global_x_min, global_x_max, global_y_min, global_y_max)

		cv2.circle(image, (center[0],center[1]), radius=2, color=(0, 0, 255), thickness=2)


		cv2.rectangle(image, (global_x_min, global_y_min), (global_x_max, global_y_max), (0, 255, 0), 2)

		cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), (0, 255, 0), 2)

		cv2.imshow('Video', image)

		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
