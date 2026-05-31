import cv2
import mediapipe as mp
import time
import random
import pandas as pd
from math import atan2, floor
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

cv2.namedWindow("VisionTouch Face Landmarker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("VisionTouch Face Landmarker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
screen_width = floor(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = floor(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
leftdx, leftdy, rightdx, rightdy = 0, 0, 0, 0
dot_x, dot_y = int(floor(random.randint(0, screen_width))), int(floor(random.randint(0, screen_height)))
data = []
# main face points for calculating face center
# 1 middle of nose bottom
# 10 middle of forehead
# 152 bottom of chin
# 234 left cheek
# 454 right cheek
FACE_LEFT_OUTER_EYE = 33
FACE_RIGHT_OUTER_EYE = 263
FACE_NOSE_BOTTOM = 1
FACE_FOREHEAD = 10
FACE_CHIN_BOTTOM = 152
FACE_LEFT_CHEEK = 234
FACE_RIGHT_CHEEK = 454
FACE_CENTER_POINTS = [FACE_NOSE_BOTTOM, FACE_FOREHEAD, FACE_CHIN_BOTTOM, FACE_LEFT_CHEEK, FACE_RIGHT_CHEEK]
FACE_CALC_POINTS = [FACE_LEFT_OUTER_EYE, FACE_RIGHT_OUTER_EYE] + FACE_CENTER_POINTS

# 468 eye right iris 473 eye left iris
LEFT_IRIS = 468
RIGHT_IRIS = 473
TRIALS_PER_ROTATION = 5
rotation = 1
def get_head_movement(landmarks):
    
    left_face = landmarks[FACE_LEFT_CHEEK]
    right_face = landmarks[FACE_RIGHT_CHEEK]
    face_width = abs(right_face.x - left_face.x)
    top_face = landmarks[FACE_FOREHEAD]
    bottom_face = landmarks[FACE_CHIN_BOTTOM]
    face_height = abs(top_face.y - bottom_face.y)
    nose = landmarks[FACE_NOSE_BOTTOM]
    face_mid_x = (left_face.x + right_face.x) / 2
    face_mid_y = (top_face.y + bottom_face.y) / 2
    face_yaw = (nose.x - face_mid_x) / face_width
    face_pitch = (nose.y - face_mid_y) / face_height
    right_outer_eye = landmarks[FACE_RIGHT_OUTER_EYE]
    left_outer_eye = landmarks[FACE_LEFT_OUTER_EYE]
    outer_eye_dy = abs(right_outer_eye.y - left_outer_eye.y)
    outer_eye_dx = abs(right_outer_eye.x - left_outer_eye.x)
    face_roll = atan2(outer_eye_dy, outer_eye_dx)
    return {"face_width": face_width, 
            "face_height": face_height, 
            "face_yaw": face_yaw, 
            "face_pitch": face_pitch,
            "face_roll": face_roll}
    

def get_face_center(landmarks):
    x_sum = 0
    y_sum = 0
    for idx in FACE_CENTER_POINTS:
        lm = landmarks[idx]
        x_sum += lm.x
        y_sum += lm.y
    center_x = x_sum / len(FACE_CENTER_POINTS)
    center_y = y_sum / len(FACE_CENTER_POINTS)
    return center_x, center_y
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb)

    time_stamp_ms = int(time.time()*1000)
    result = landmarker.detect_for_video(mp_image, time_stamp_ms)
    
    if result.face_landmarks:
        # first face only
        landmarks = result.face_landmarks[0]
        # place all the landmarks as green dots
        for idx in FACE_CALC_POINTS:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Example eye/iris points
        important_points = [LEFT_IRIS, RIGHT_IRIS]
        # place all the important landmarks as blue dots
        for idx in important_points:
            lm = landmarks[idx]
            if(idx == LEFT_IRIS):
                leftdx, leftdy = lm.x, lm.y
            elif(idx == RIGHT_IRIS):
                rightdx, rightdy = lm.x, lm.y
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    
    #place the random dot on the screen
    cv2.circle(frame, (dot_x, dot_y), 8, (0, 0, 0), -1)
    # place the face center on the screen
    face_center_x, face_center_y = get_face_center(landmarks)
    cv2.circle(frame, (int(face_center_x * w), int(face_center_y * h)), 8, (255, 0, 0), -1)
    cv2.putText(frame, f"Sample: {floor(rotation / TRIALS_PER_ROTATION) + 1} Trial: {rotation}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("VisionTouch Face Landmarker", frame)
    key = cv2.waitKey(1)
    # only hit when dot does not show up in view
    if key == ord("r"):
        dot_x, dot_y = random.randint(0, screen_width), random.randint(0, screen_height)
    if key == ord(" "):
        head = get_head_movement(landmarks)
        data.append({
            "leftdx": leftdx,
            "leftdy": leftdy,
            "rightdx": rightdx,
            "rightdy": rightdy,
            "face_center_x": face_center_x,
            "face_center_y": face_center_y,
            "face_width": head["face_width"],
            "face_height": head["face_height"],
            "face_yaw": head["face_yaw"],
            "face_pitch": head["face_pitch"],
            "face_roll": head["face_roll"],
            "dot_x": dot_x / screen_width,
            "dot_y": dot_y / screen_height
        })
        if(rotation % TRIALS_PER_ROTATION == 0):
            dot_x, dot_y = random.randint(0, screen_width), random.randint(0, screen_height)
        rotation+=1 
    if key & 0xFF == ord("q"):
        break
df = pd.DataFrame(data)
df.to_csv("calibration_data.csv", index=False)
print(df)
cap.release()
landmarker.close()
cv2.destroyAllWindows()