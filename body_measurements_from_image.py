import cv2
import mediapipe as mp 
import math 
import os

# __________________________________________________________________________________________________________

# Function for the entire program (To be done later)
""" def getMeasurements(image_path):
    image = cv2.imread(image_path, 1) """
# __________________________________________________________________________________________________________

# Text file for storing all the calculated measurements of the body

report_file_path = "C:\\Users\\OMOLP094\\Desktop\\GitHub Repos Of Projects\\Body-Measurements-From-Image\\Body Measurements Report.txt"  # Path where the Child's body measurements report has to be saved 

# Check if the file already exists (It could be the Body measurements report of the previous user)
if os.path.exists(report_file_path):
    # Delete the existing body measurements report
    os.remove(report_file_path)
    print(f"Body Measurements Report : '{report_file_path}' deleted.")

# Create a New Body measurements report file for the new user 
report_file =  open(report_file_path, 'w')
print(f"Body Measurements Report at '{report_file_path}' created and opened in write mode.")

# Functions for calculating Euclidean Distances in 2D Space

def calculate_2d_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Functions for calculating Euclidean Distances in 3D Space
def calculate_3d_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

#________________Obtaining eye width in cms for the user with the age (months) he/she provides______________

# A dictionary containing the horizontal diameter of the visible part of the eye, often known as the corneal diameter for people of different age groups in centimetres

# Keys are age groups in months and values are average eye width for the age groups in centimetres

eye_width_averages_all_ages_cms = {
    # From the below given age groups, difference in eye width between consecutive age groups has been taken as 0.2 cms
    "0-3 months": 1.6, # average eye width in cms for age 0 months to 3 months (including 0 month but excluding 3 months)
    "3-6 months": 2.0, # average eye width in cms for age 3 months to 6 months (including 3 months but excluding 6 months), similar for others below
    "6-12 months": 2.4,
    "12-24 months": 2.6,
    "24-36 months": 2.8,
    "36-72 months": 3.0,
    # From the below given age groups, difference in eye width between consecutive age groups has been taken as 0.3 cms
    "72-120 months": 3.2,
    "120-156 months": 3.5,
    "156-216 months": 3.8,
    "216-1800 months": 4.1
}

user_name = input("Enter the name of the child : ")
report_file.write(f"Child's Name : {user_name} \n\n")

# Obtaining the user's age in months 
print("Upload an Image of the Baby that has been captured from a Distance of 1 to 1.5 metres.")
user_age_months = 0
months_or_years = input("Enter the unit in which you want to provide your age (Months or Years) : ")
user_age_input = int(input(f"Enter your Age in {months_or_years.title()}, Just enter the number. For e.g. If your age is 8 months, only enter 8 and If your age is 6 years, only enter 6 : "))
if months_or_years.lower() == "months":
    user_age_months = user_age_input
    report_file.write(f"Child's Age In Months : {user_age_months} \n\n")
elif months_or_years.lower() == "years":
    user_age_months = user_age_input * 12
    report_file.write(f"Child's Age In Months : {user_age_months} \n\n")
else:
    print("Enter your age either in Months or Years, Please don't enter the age value in any other units.")



# Conditional ladder to get the average eye width of a person belonging to a particular age group (months)

if 0 <= user_age_months and user_age_months < 3: # 0 months to 3 months (including 0 month but excluding 3 months)
    user_eye_width_cms = eye_width_averages_all_ages_cms['0-3 months']
elif 3 <= user_age_months and user_age_months < 6: # 3 months to 6 months (including 3 months but excluding 6 months), similar for others below
    user_eye_width_cms = eye_width_averages_all_ages_cms['3-6 months']
elif 6 <= user_age_months and user_age_months < 12:
    user_eye_width_cms = eye_width_averages_all_ages_cms['6-12 months']
elif 12 <= user_age_months and user_age_months < 24:
    user_eye_width_cms = eye_width_averages_all_ages_cms['12-24 months']   
elif 24 <= user_age_months and user_age_months < 36:
    user_eye_width_cms = eye_width_averages_all_ages_cms['24-36 months']  
elif 36 <= user_age_months and user_age_months < 72:
    user_eye_width_cms = eye_width_averages_all_ages_cms['36-72 months']  
elif 72 <= user_age_months and user_age_months < 120:
    user_eye_width_cms = eye_width_averages_all_ages_cms['72-120 months'] 
elif 120 <= user_age_months and user_age_months < 156:
    user_eye_width_cms = eye_width_averages_all_ages_cms['120-156 months'] 
elif 156 <= user_age_months and user_age_months < 216:
    user_eye_width_cms = eye_width_averages_all_ages_cms['156-216 months'] 
elif 216 <= user_age_months and user_age_months < 1800:
    user_eye_width_cms = eye_width_averages_all_ages_cms['216-1800 months']
else:
    print("Enter a Valid age which is in between 0 Years to 150 Years") 

print(f"Your Eye Width in Centimetres is {user_eye_width_cms}")


# _______________________Mediapipe imports, objects & settings__________________________________

# Mediapipe imports & settings

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

drawing_style_for_lms = mp_drawing.DrawingSpec(color = (0,255,0), thickness = 2, circle_radius = 2)
drawing_style_for_connections_of_lms = mp_drawing.DrawingSpec(color = (255,0,0), thickness = 2, circle_radius = 2)

holistic = mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

# Getting the image

image = cv2.imread("C:\\Users\\OMOLP094\\Desktop\\GitHub Repos Of Projects\\Body-Measurements-From-Image\\test_images\\baby_test_image_1.jpg", cv2.IMREAD_UNCHANGED)

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

# To print all the connections between pose landmarks 
# print(mp_holistic.POSE_CONNECTIONS)

# _____________________________Image width & height in pixels________________________________________________

# Getting the image dimensions in pixels

img_height_px, img_width_px, channels = image.shape

# Other option for getting the dimensions of the image
# height = image.shape[0]
# width = image.shape[1]
# channels = image.shape[2]

# print(f"Image height in pixels : {img_height_px}")
# print(f"Image width in pixels : {img_width_px}")

# ______Number of cms in 1 pixel using the eye width in pixels & cms for the user whose age is provided______

# Calculating the distance between the left & right endpoints of the right eye in pixels by using the landmarks obtained for them in the face landmarks

# print(face_landmarks)

# Right End Of The Right Eye

# Extract coordinates of landmark 130 from the face landmarks which represents the right end of the right eye 

right_end_right_eye_lm_coords = face_landmarks.landmark[130]
right_end_right_eye_lm_coord_x, right_end_right_eye_lm_coord_y, right_end_right_eye_lm_coord_z = right_end_right_eye_lm_coords.x, right_end_right_eye_lm_coords.y, right_end_right_eye_lm_coords.z

# Converting the normalized values of x & y coordinates to standard values by multiplying them with image width & height in pixels to scale them & then converting those values into integers, because we would need integers to calculate the distances between points using math.dist() and draw lines connecting the landmarks using cv2.line

right_end_right_eye_lm_coord_x_px = int(right_end_right_eye_lm_coord_x * img_width_px)
right_end_right_eye_lm_coord_y_px = int(right_end_right_eye_lm_coord_y * img_height_px)

right_end_right_eye_lm_coords_only_x_and_y = [right_end_right_eye_lm_coord_x_px, right_end_right_eye_lm_coord_y_px] # This is the variable that has the position of the right end of the right eye that has to be used

# print(f'Right End Of The Right Eye Landmark Coordinates - X : {right_end_right_eye_lm_coord_x}, Y: {right_end_right_eye_lm_coord_y}, Z: {right_end_right_eye_lm_coord_z}')

# print(f"Position (x & y Coordinates) of the Right end of the Right Eye : {right_end_right_eye_lm_coords_only_x_and_y}")

# Drawing a circle of red color on the Right end of the Right eye in the image
cv2.circle(image_BGR, right_end_right_eye_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)


# Left End Of The Right Eye

# Extract coordinates of landmark 243 from the face landmarks which represents the left end of the right eye
 
left_end_right_eye_lm_coords = face_landmarks.landmark[243]
left_end_right_eye_lm_coord_x, left_end_right_eye_lm_coord_y, left_end_right_eye_lm_coord_z = left_end_right_eye_lm_coords.x, left_end_right_eye_lm_coords.y, left_end_right_eye_lm_coords.z

# Converting the normalized values of x & y coordinates to standard values by multiplying them with image width & height in pixels to scale them & then converting those values into integers, because we would need integers to calculate the distances between points using math.dist() and draw lines connecting the landmarks using cv2.line

left_end_right_eye_lm_coord_x_px = int(left_end_right_eye_lm_coord_x * img_width_px)
left_end_right_eye_lm_coord_y_px = int(left_end_right_eye_lm_coord_y * img_height_px)

left_end_right_eye_lm_coords_only_x_and_y = [left_end_right_eye_lm_coord_x_px, left_end_right_eye_lm_coord_y_px]  # This is the variable that has the position of the left end of the right eye that has to be used

# print(f'Left End Of The Right Eye Landmark Coordinates - X : {left_end_right_eye_lm_coord_x}, Y: {left_end_right_eye_lm_coord_y}, Z: {left_end_right_eye_lm_coord_z}')

# print(f"Position (x & y Coordinates) of the Left end of the Right Eye : {left_end_right_eye_lm_coords_only_x_and_y}")

# Drawing a circle of red color on the Right end of the Right eye in the image
cv2.circle(image_BGR, left_end_right_eye_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Drawing a line joining the Right end & Left end landmarks of the Right eye
cv2.line(image_BGR, right_end_right_eye_lm_coords_only_x_and_y, left_end_right_eye_lm_coords_only_x_and_y, (0, 255, 0), 3) 

# ________Euclidean distance between Right end & Left end landmarks of the Right eye_____________________

# Calculating Euclidean distance between Right end & Left end landmarks of the Right eye, this would give us the width of the user's eye in pixels

# Same values are being obtained from all two approaches that have been used below for calculating Euclidean distance between 2 points on the image

# user_eye_width_px_math_2d = math.dist(right_end_right_eye_lm_coords_only_x_and_y, left_end_right_eye_lm_coords_only_x_and_y) # calculating eye width in pixels (distance between 2 landmarks in pixels) using dist function from math module

# print(f"user_eye_width_px_math_2d : {user_eye_width_px_math_2d}")

# user_eye_width_px_custom_2d = calculate_2d_distance(right_end_right_eye_lm_coords_only_x_and_y, left_end_right_eye_lm_coords_only_x_and_y) # calculating eye width in pixels (distance between 2 landmarks in pixels) using custom function for 2D space

# print(f"user_eye_width_px_custom_2d : {user_eye_width_px_custom_2d}")

# So, I am using the calculate_2d_distance function that I've created above for calculating Euclidean distance between 2 points on the image

user_eye_width_px = math.dist(right_end_right_eye_lm_coords_only_x_and_y, left_end_right_eye_lm_coords_only_x_and_y) # calculating eye width in pixels (distance between 2 landmarks in pixels) using custom function for 2D space

print(f"Eye width of the user in pixels : {user_eye_width_px}")

# ________________________________________________Pixel To Centimetres Conversion factor_____________________________________________

# Now obtaining the number of centimetres in one pixel (i.e, the conversion factor for converting distances obtained in pixels into centimetres) 

num_of_cms_in_one_pixel = user_eye_width_cms / user_eye_width_px

print(f"Number of centimetres in one pixel for the uploaded image is {num_of_cms_in_one_pixel}")

# _____________________Obtaining Head Circumference of the User in Centimetres__________________________

# Extracting coordinates of Left end (Left ear) & Right end (Right ear) landmarks of the face, distance between these 2 landmarks is the diameter of the head

# Obtaining the landmarks of the Left & Right end of the face (will be used as Head Diameter) 

# Obtaining the x & y coordinates in standard values of the landmark of the right end of the face (right ear)

right_end_face_coords = pose_landmarks.landmark[8]  # Index 8 corresponds to the right ear

right_end_face_coord_x, right_end_face_coord_y, right_end_face_coord_z = right_end_face_coords.x, right_end_face_coords.y, right_end_face_coords.z

right_end_face_coord_x_px, right_end_face_coord_y_px = int(right_end_face_coord_x * img_width_px), int(right_end_face_coord_y * img_height_px)

right_end_face_lm_coords_only_x_and_y = [right_end_face_coord_x_px, right_end_face_coord_y_px]

cv2.circle(image_BGR, right_end_face_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Obtaining the x & y coordinates in standard values of the landmark of the left end of the face (left ear)
left_end_face_coords = pose_landmarks.landmark[7]  # Index 7 corresponds to the left ear

left_end_face_coord_x, left_end_face_coord_y, left_end_face_coord_z = left_end_face_coords.x, left_end_face_coords.y, left_end_face_coords.z

left_end_face_coord_x_px, left_end_face_coord_y_px = int(left_end_face_coord_x * img_width_px), int(left_end_face_coord_y * img_height_px)

left_end_face_lm_coords_only_x_and_y = [left_end_face_coord_x_px, left_end_face_coord_y_px]

cv2.circle(image_BGR, left_end_face_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Drawing a line joining the Right end & Left end landmarks of the Face
cv2.line(image_BGR, right_end_face_lm_coords_only_x_and_y, left_end_face_lm_coords_only_x_and_y, (0, 255, 0), 3) 

# Calculating User Head Diameter in pixels 

user_head_diameter_px = calculate_2d_distance(right_end_face_lm_coords_only_x_and_y, left_end_face_lm_coords_only_x_and_y) 

# Calculating User Head Diameter in centimetres using pixel to centimetre conversion factor

user_head_diameter_cms = user_head_diameter_px * num_of_cms_in_one_pixel

print(f"User Head Diameter in Centimetres : {user_head_diameter_cms}")

# Calculating User Head Circumference in centimetres using User Head Diameter value in Centimetres

user_head_circumference_cms = math.pi * user_head_diameter_cms

print(f"User Head Circumference in Centimetres : {user_head_circumference_cms}")

report_file.write(f"Child's Head Circumference in Centimetres : {user_head_circumference_cms} \n\n")

# __________________________Obtaining Arm Length of the User in Centimetres__________________________________

# Extracting coordinates of Left Shoulder & Left Wrist landmarks, distance between these 2 landmarks is the length of the Left Arm

# Obtaining the x & y coordinates in standard values of the landmark of the Left Shoulder

left_shoulder_coords = pose_landmarks.landmark[11]  # Index 11 corresponds to the left shoulder

left_shoulder_coord_x, left_shoulder_coord_y, left_shoulder_coord_z = left_shoulder_coords.x, left_shoulder_coords.y, left_shoulder_coords.z

left_shoulder_coord_x_px, left_shoulder_coord_y_px = int(left_shoulder_coord_x * img_width_px), int(left_shoulder_coord_y * img_height_px)

left_shoulder_lm_coords_only_x_and_y = [left_shoulder_coord_x_px, left_shoulder_coord_y_px]

cv2.circle(image_BGR, left_shoulder_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Obtaining the x & y coordinates in standard values of the landmark of the Left Wrist

left_wrist_coords = pose_landmarks.landmark[15]     # Index 15 corresponds to the left wrist

left_wrist_coord_x, left_wrist_coord_y, left_wrist_coord_z = left_wrist_coords.x, left_wrist_coords.y, left_wrist_coords.z

left_wrist_coord_x_px, left_wrist_coord_y_px = int(left_wrist_coord_x * img_width_px), int(left_wrist_coord_y * img_height_px)

left_wrist_lm_coords_only_x_and_y = [left_wrist_coord_x_px, left_wrist_coord_y_px]

cv2.circle(image_BGR, left_wrist_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Obtaining the x & y coordinates in standard values of the landmark of the tip of the middle finger of the left hand
'''
left_mid_finger_tip_coords = pose_landmarks.landmark[12]     # Index 15 corresponds to the left wrist

left_mid_finger_tip_coord_x, left_mid_finger_tip_coord_y, left_mid_finger_tip_coord_z = left_mid_finger_tip_coords.x, left_mid_finger_tip_coords.y, left_mid_finger_tip_coords.z

left_mid_finger_tip_coord_x_px, left_mid_finger_tip_coord_y_px = int(left_mid_finger_tip_coord_x * img_width_px), int(left_mid_finger_tip_coord_y * img_height_px)

left_mid_finger_tip_lm_coords_only_x_and_y = [left_mid_finger_tip_coord_x_px, left_mid_finger_tip_coord_y_px]

cv2.circle(image_BGR, left_mid_finger_tip_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)
'''
# Drawing a line joining the Left Shoulder & Left Wrist landmarks 
cv2.line(image_BGR, left_shoulder_lm_coords_only_x_and_y, left_wrist_lm_coords_only_x_and_y, (0, 255, 0), 3) 

# Calculating User Arm Length in pixels 

user_arm_length_px = calculate_2d_distance(left_shoulder_lm_coords_only_x_and_y, left_wrist_lm_coords_only_x_and_y) 

# Calculating User Arm Length in centimetres using pixel to centimetre conversion factor

user_arm_length_cms = user_arm_length_px * num_of_cms_in_one_pixel

print(f"User Arm Length (Distance between Shoulder & Wrist) in Centimetres : {user_arm_length_cms}")

report_file.write(f"Child's Arm Length in Centimetres {user_arm_length_cms} \n\n")

# ____________________________Obtaining the Height of the User in Centimetres________________________________

# Extracting coordinates of the Top Head Landmark, Left Ankle & Right Ankle landmarks, distance between the top head landmark & left ankle landmark or right ankle landmark gives us the height

head_top_coords = face_landmarks.landmark[10] # Index 10 corresponds to the top of the head
left_ankle_coords = pose_landmarks.landmark[27] # Index 27 corresponds to the left ankle
right_ankle_coords = pose_landmarks.landmark[28] # Index 28 corresponds to the right ankle

# Obtaining the x & y coordinates in standard values of the landmark of the Top of the Head

head_top_coord_x, head_top_coord_y, head_top_coord_z = head_top_coords.x, head_top_coords.y, head_top_coords.z

head_top_coord_x_px, head_top_coord_y_px = int(head_top_coord_x * img_width_px), int(head_top_coord_y * img_height_px)

head_top_lm_coords_only_x_and_y = [head_top_coord_x_px, head_top_coord_y_px]

cv2.circle(image_BGR, head_top_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Obtaining the x & y coordinates in standard values of the landmark of the Right Ankle

right_ankle_coord_x, right_ankle_coord_y, right_ankle_coord_z = right_ankle_coords.x, right_ankle_coords.y, right_ankle_coords.z

right_ankle_coord_x_px, right_ankle_coord_y_px = int(right_ankle_coord_x * img_width_px), int(right_ankle_coord_y * img_height_px)

right_ankle_lm_coords_only_x_and_y = [right_ankle_coord_x_px, right_ankle_coord_y_px]

cv2.circle(image_BGR, right_ankle_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Obtaining the x & y coordinates in standard values of the landmark of the Left Ankle

left_ankle_coord_x, left_ankle_coord_y, left_ankle_coord_z = left_ankle_coords.x, left_ankle_coords.y, left_ankle_coords.z

left_ankle_coord_x_px, left_ankle_coord_y_px = int(left_ankle_coord_x * img_width_px), int(left_ankle_coord_y * img_height_px)

left_ankle_lm_coords_only_x_and_y = [left_ankle_coord_x_px, left_ankle_coord_y_px]

cv2.circle(image_BGR, left_ankle_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)


# Drawing a line joining the Top of the head & Right ankle landmarks 
cv2.line(image_BGR, head_top_lm_coords_only_x_and_y, right_ankle_lm_coords_only_x_and_y, (0, 255, 0), 3)

# Drawing a line joining the Top of the head & Left ankle landmarks 
cv2.line(image_BGR, head_top_lm_coords_only_x_and_y, left_ankle_lm_coords_only_x_and_y, (0, 255, 0), 3)


# Calculating the distance between top of the head and right ankle in pixels 

distance_head_top_right_ankle_px = calculate_2d_distance(head_top_lm_coords_only_x_and_y, right_ankle_lm_coords_only_x_and_y) 

# Calculating the distance between top of the head and right ankle in centimetres using pixel to centimetre conversion factor

distance_head_top_right_ankle_cms = distance_head_top_right_ankle_px * num_of_cms_in_one_pixel

print(f"Distance between the Top of the Head and Right Ankle in Centimetres : {distance_head_top_right_ankle_cms}")


# Calculating the distance between top of the head and left ankle in pixels 

distance_head_top_left_ankle_px = calculate_2d_distance(head_top_lm_coords_only_x_and_y, left_ankle_lm_coords_only_x_and_y) 

# Calculating the distance between top of the head and left ankle in centimetres using pixel to centimetre conversion factor

distance_head_top_left_ankle_cms = distance_head_top_left_ankle_px * num_of_cms_in_one_pixel

print(f"Distance between the Top of the Head and Left Ankle in Centimetres : {distance_head_top_left_ankle_cms}")


# Calculating the User Height in Centimetres by taking the Average of Distance between the Top of the Head and Right Ankle in Centimetres & Distance between the Top of the Head and Left Ankle in Centimetres

user_height_cms = (distance_head_top_right_ankle_cms + distance_head_top_left_ankle_cms) / 2

print(f"User Height in Centimetres : {user_height_cms}")
print("The value of Height can be 0 to 15 cms less than the Actual Height of the Baby or Person whose Image has been provided.")

report_file.write(f"Child's Height in Centimetres : {user_height_cms} \n\n")

# Displaying the path of the file containing the Child's Body measurements report
print(f"Child's Body Measurements Report Saved in {report_file.name}")

report_file.close()

# _______________Obtaining Arm Circumference in Centimetres (Not working due to unavailability of Wrist Landmarks (Only 1 landmark for wrist centre is available in Mediapipe)__________________________
    
# Obtaining the x & y coordinates in standard values of the landmark of the Left Wrist
left_wrist_coords = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value]

# print(f"Coordinates of the Left Wrist Landmark : {left_wrist_coords}")

left_wrist_coord_x, left_wrist_coord_y, left_wrist_coord_z = left_wrist_coords.x, left_wrist_coords.y, left_wrist_coords.z

left_wrist_coord_x_px, left_wrist_coord_y_px = int(left_wrist_coord_x * img_width_px), int(left_wrist_coord_y * img_height_px)

left_wrist_lm_coords_only_x_and_y = [left_wrist_coord_x_px, left_wrist_coord_y_px]

cv2.circle(image_BGR, left_wrist_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Obtaining the x & y coordinates in standard values of the landmark of the Left Wrist

right_wrist_coords = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value]

# print(f"Coordinates of the Right Wrist Landmark : {right_wrist_coords}")

right_wrist_coord_x, right_wrist_coord_y, right_wrist_coord_z = right_wrist_coords.x, right_wrist_coords.y, right_wrist_coords.z

right_wrist_coord_x_px, right_wrist_coord_y_px = int(right_wrist_coord_x * img_width_px), int(right_wrist_coord_y * img_height_px)

right_wrist_lm_coords_only_x_and_y = [right_wrist_coord_x_px, right_wrist_coord_y_px]

cv2.circle(image_BGR, right_wrist_lm_coords_only_x_and_y, 2, (0, 0, 255), thickness=cv2.FILLED)

# Drawing a line joining the left & right wrist landmarks 
# cv2.line(image_BGR, left_wrist_lm_coords_only_x_and_y, right_wrist_lm_coords_only_x_and_y, (0, 255, 0), 3)

# Obtaining Wrist Width or Diameter which will be taken as the Arm Diameter or Width too for finding Arm Circumference

# __________________________________Closing Holistic Model___________________________________________

holistic.close()

# _______________Displaying the Uploaded Image with Landmarks & Line joining the Landmarks________________

# For Resizing the BGR image with landmarks & connections drawn on it before displaying it in the OpenCV window (This is only for Images with High Resolution which are not completely fitting in the OpenCV window)
# image_BGR_resized = cv2.resize(image_BGR, (0, 0), fx = 0.5, fy = 0.5)

# Create a window with a specific name
cv2.namedWindow('Body Measurements With Image', cv2.WINDOW_NORMAL)

# Resize the window to a specific width and height
cv2.resizeWindow('Body Measurements With Image', 800, 600)

# Display the image in the resized window
cv2.imshow('Body Measurements With Image', image_BGR)

# cv2.imshow("Body Measurements Window", image_BGR_resized)
# cv2.imshow("Body Measurements Window", image_BGR)

k = cv2.waitKey(0)
if k == ord('q'):
    cv2.destroyAllWindows()

# _______________________________________________________________________________________________________
