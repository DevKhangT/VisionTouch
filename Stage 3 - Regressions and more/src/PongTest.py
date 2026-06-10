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
frame_width = floor(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = floor(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame Width: {frame_width}, Frame Height: {frame_height}")

name = input("What is this experiments name?")
FIXED_SPEED = 4
FIXED_RADIUS = 10
FIXED_MIN = 3
FIXED_SECS = FIXED_MIN * 60
FIXED_MSECS = FIXED_SECS * 1000
radius = FIXED_RADIUS
leftdx, leftdy, rightdx, rightdy = 0, 0, 0, 0
dot_x, dot_y = random.randint(radius+1, frame_width-radius-1), random.randint(radius+1, frame_height-radius-1)

dot_vx = FIXED_SPEED
dot_vy = dot_vx * (frame_height / frame_width)
frame_id = 0
experiment = False

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
    valid_face_detected = bool(result.face_landmarks)
    if valid_face_detected:
        
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
            # place the face center on the screen
        face_center_x, face_center_y = get_face_center(landmarks)
        cv2.circle(frame, (int(face_center_x * w), int(face_center_y * h)), 8, (255, 0, 0), -1)
    
    if experiment and valid_face_detected:
        cv2.putText(frame, f"Frame ID: {frame_id}, Speed: {FIXED_SPEED}, Radius: {FIXED_RADIUS}, Time (ms): {time_stamp_ms}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # if collides with walls
        dot_x += dot_vx
        dot_y += dot_vy
        collide_x = (dot_x - radius <= 0 or dot_x + radius >= frame_width)
        collide_y = (dot_y - radius <= 0 or dot_y + radius >= frame_height)
        if(collide_x):
            dot_vx = -dot_vx
            dot_x = max(radius, min(frame_width-radius, dot_x))
        if(collide_y):
            dot_vy = -dot_vy
            dot_y = max(radius, min(frame_height-radius, dot_y))
        #place the dot
        cv2.circle(frame, (int(dot_x), int(dot_y)), radius, (0, 0, 255), -1)
        head = get_head_movement(landmarks)
        data.append({
            "frame_id": frame_id,
            "path_id": "pong constant velocity",
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
            "dot_x": dot_x / frame_width,
            "dot_y": dot_y / frame_height,
            "dot_vx": dot_vx,
            "dot_vy": dot_vy,
            "current_timestamp": time_stamp_ms,
            "elapsed_time": time.time() - experiment_start_time,
            "collide_x": int(collide_x),
            "collide_y": int(collide_y),
            "is_colliding": int(collide_x or collide_y),
            "experiment_name": name,
            "path_id": "pong_constant_velocity",
            "fixed_speed": FIXED_SPEED,
            "radius": radius,
            "valid_face_detected": int(valid_face_detected)
        })
        frame_id += 1

    cv2.imshow("VisionTouch Face Landmarker", frame)
    key = cv2.waitKey(1)
    if key == ord(" ") and not experiment:
        experiment = True
        experiment_start_time = time.time()
        frame_id = 0
        data = []
    if key & 0xFF == ord("q") or (experiment and time.time() - experiment_start_time > FIXED_SECS):
        experiment = False
        break
df = pd.DataFrame(data)
df.to_csv(f"PingPong_{name}.csv", index=False)
print(df)
cap.release()
landmarker.close()
cv2.destroyAllWindows()