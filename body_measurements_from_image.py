import cv2
import mediapipe as mp 
import math # For calculating Euclidean distance between landmarks

# _____________________________________________________
# Function for the entire program (To be done later)
""" def getMeasurements(image_path):
    image = cv2.imread(image_path, 1) """
# ______________________________________________________

# Mediapipe imports & settings

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

drawing_style_for_lms = mp_drawing.DrawingSpec(color = (0,255,0), thickness = 2, circle_radius = 2)
drawing_style_for_connections_of_lms = mp_drawing.DrawingSpec(color = (255,0,0), thickness = 2, circle_radius = 2)

holistic = mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

# Getting the image

image = cv2.imread("C:\\Users\\OMOLP094\\Desktop\\GitHub Repos Of Projects\\Body-Measurements-From-Image\\test_images\\baby_test_image_straight.png", cv2.IMREAD_UNCHANGED)

# Converting the loaded image to RGB to make the detections in the image using mediapipe holistic
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Getting the coordinates of the landmarks for the image using holistic
body_lm_coords_using_holistic = holistic.process(image_RGB)

# Converting the RGB image back to it's original BGR format after getting the coordinates of all the landmarks using mediapipe holistic, We'll draw all the landmarks detected on this original BGR form of the image
image_BGR = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)

# Storing all the landmarks in different variables

face_landmarks = body_lm_coords_using_holistic.face_landmarks

right_hand_landmarks = body_lm_coords_using_holistic.right_hand_landmarks

left_hand_landmarks = body_lm_coords_using_holistic.left_hand_landmarks

pose_landmarks = body_lm_coords_using_holistic.pose_landmarks

# Drawing all the detected landmarks on the image

# mp_drawing.draw_landmarks(image_BGR, face_landmarks, mp_holistic.FACEMESH_TESSELATION, drawing_style_for_lms, drawing_style_for_connections_of_lms)

# mp_drawing.draw_landmarks(image_BGR, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_style_for_lms, drawing_style_for_connections_of_lms)

# mp_drawing.draw_landmarks(image_BGR, left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_style_for_lms, drawing_style_for_connections_of_lms)

# mp_drawing.draw_landmarks(image_BGR, pose_landmarks, mp_holistic.POSE_CONNECTIONS, drawing_style_for_lms, drawing_style_for_connections_of_lms)

# Getting the image dimensions in pixels

img_height_px, img_width_px, channels = image.shape

# Other option for getting the dimensions of the image
# height = image.shape[0]
# width = image.shape[1]
# channels = image.shape[2]

print(f"Image height in pixels : {img_height_px}")
print(f"Image width in pixels : {img_width_px}")

# For Resizing the BGR image with landmarks & connections drawn on it before displaying it in the OpenCV window (This is only for Images with High Resolution which are not completely fitting in the OpenCV window)
# image_BGR_resized = cv2.resize(image_BGR, (0,0), fx = 0.2, fy = 0.2)

# Calculating the distance between the left & right endpoints of the right eye in pixels by using the landmarks obtained for them in the face landmarks
print(face_landmarks)
# left_end_right_eye_lm_coords = face_landmarks[130]
# right_end_right_eye_lm_coords = face_landmarks[243]

# print(f"Coordinates of the left end of the right eye : {left_end_right_eye_lm_coords}")
# print(f"Coordinates of the right end of the right eye : {right_end_right_eye_lm_coords}")

# Extract coordinates of landmark 130 from the face landmarks

right_end_right_eye_lm_coords = face_landmarks.landmark[130]
right_end_right_eye_lm_coord_x, right_end_right_eye_lm_coord_y, right_end_right_eye_lm_coord_z = right_end_right_eye_lm_coords.x, right_end_right_eye_lm_coords.y, right_end_right_eye_lm_coords.z

# Convert normalized coordinates to pixel values & then converting those pixel values into integers, because we would need integers to calculate the distances between points
right_end_right_eye_lm_coord_x_px = int(right_end_right_eye_lm_coord_x * img_width_px)
right_end_right_eye_lm_coord_y_px = int(right_end_right_eye_lm_coord_y * img_height_px)

right_end_right_eye_lm_coords_only_x_and_y = [right_end_right_eye_lm_coord_x_px, right_end_right_eye_lm_coord_y_px]

print(f"Right end of right eye landmarks x & y coords : {right_end_right_eye_lm_coords_only_x_and_y}")

# Extract coordinates of landmark 243 from the face landmarks
left_end_right_eye_lm_coords = face_landmarks.landmark[243]
left_end_right_eye_lm_coord_x, left_end_right_eye_lm_coord_y, left_end_right_eye_lm_coord_z = left_end_right_eye_lm_coords.x, left_end_right_eye_lm_coords.y, left_end_right_eye_lm_coords.z

# Convert normalized coordinates to pixel values & then converting those pixel values into integers, because we would need integers to calculate the distances between points
left_end_right_eye_lm_coord_x_px = int(left_end_right_eye_lm_coord_x * img_width_px)
left_end_right_eye_lm_coord_y_px = int(left_end_right_eye_lm_coord_y * img_height_px)

left_end_right_eye_lm_coords_only_x_and_y = [left_end_right_eye_lm_coord_x_px, left_end_right_eye_lm_coord_y_px]

print(f"Left end of right eye landmarks x & y coords : {left_end_right_eye_lm_coords_only_x_and_y}")

# print(f'right_end_right_eye_lm_coords - X : {right_end_right_eye_lm_coord_x}, Y: {right_end_right_eye_lm_coord_y}, Z: {right_end_right_eye_lm_coord_z}')

# print(f'left_end_right_eye_lm_coords - X : {left_end_right_eye_lm_coord_x}, Y: {left_end_right_eye_lm_coord_y}, Z: {left_end_right_eye_lm_coord_z}')

cv2.line(image_BGR, right_end_right_eye_lm_coords_only_x_and_y, left_end_right_eye_lm_coords_only_x_and_y, (0, 200, 0), 3) 

cv2.imshow("Body Measurements Window", image_BGR)

k = cv2.waitKey(0)
if k == ord('q'):
    cv2.destroyAllWindows()




