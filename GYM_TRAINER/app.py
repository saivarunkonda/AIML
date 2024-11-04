import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate("C:\\Users\\Samarth D Gothe\\OneDrive\\Documents\\Anaconda\\workout.json")  # Update the path
firebase_app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://workout-a1b41-default-rtdb.firebaseio.com/'
}, name='workout-app-59')

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define constants
BICEP_CURL_ANGLE_UP_THRESHOLD = 30
BICEP_CURL_ANGLE_DOWN_THRESHOLD = 160
LATERAL_RAISE_ANGLE_UP_THRESHOLD = 100
LATERAL_RAISE_ANGLE_DOWN_THRESHOLD = 20

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

def perform_action(action_name, angle_function, up_threshold, down_threshold):
    cap = cv2.VideoCapture('http://192.168.38.172:81/stream')  # Update with correct stream URL
    
    if not cap.isOpened():
        print(f"Failed to open the video stream for {action_name}")
        return
    
    counter = 0
    stage = None
    buffer = False
    counter_ref = db.reference(f'{action_name}/counter', app=firebase_app)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    print("Failed to grab frame or empty frame received")
                    continue
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Angle-specific function
                    angle = angle_function(landmarks)
                    
                    if angle:
                        cv2.putText(image, str(angle), 
                                    tuple(np.multiply([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                                                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y], [640, 480]).astype(int)),  
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        if action_name == "bicep_curl":
                            if angle < BICEP_CURL_ANGLE_UP_THRESHOLD:
                                stage = "up"
                                buffer = False
                            elif angle > BICEP_CURL_ANGLE_DOWN_THRESHOLD and stage == 'up' and not buffer:
                                stage = "down"
                                counter += 1
                                buffer = True
                                try:
                                    counter_ref.set(counter)
                                    print(f"{action_name.capitalize()} counter updated to {counter}")
                                except Exception as e:
                                    print(f"Failed to update Firebase: {e}")
                            elif angle < BICEP_CURL_ANGLE_DOWN_THRESHOLD and buffer:
                                buffer = False
                        elif action_name == "lateral_raise":
                            if angle < LATERAL_RAISE_ANGLE_DOWN_THRESHOLD:
                                stage = "down"
                                buffer = False
                            elif angle > LATERAL_RAISE_ANGLE_UP_THRESHOLD and stage == 'down' and not buffer:
                                stage = "up"
                                counter += 1
                                buffer = True
                                try:
                                    counter_ref.set(counter)
                                    print(f"{action_name.capitalize()} counter updated to {counter}")
                                except Exception as e:
                                    print(f"Failed to update Firebase: {e}")
                            elif angle > LATERAL_RAISE_ANGLE_DOWN_THRESHOLD and buffer:
                                buffer = False
                        
                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                cv2.putText(image, 'REPS', (15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'STAGE', (65, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (60, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                cv2.imshow(f'{action_name} Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    counter_ref.set(0)
                    print(f"{action_name.capitalize()} counter reset in Firebase")
                    break
    
        except Exception as e:
            print(f"Error in {action_name} capture loop: {e}")
    
        finally:
            cap.release()
            cv2.destroyAllWindows()

def left_hip_action():
    def lateral_raise_angle(landmarks):
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        return calculate_angle(wrist, shoulder, hip)
    
    perform_action("lateral_raise", lateral_raise_angle, LATERAL_RAISE_ANGLE_UP_THRESHOLD, LATERAL_RAISE_ANGLE_DOWN_THRESHOLD)

def write_text():
    def bicep_curl_angle(landmarks):
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        return calculate_angle(shoulder, elbow, wrist)
    
    perform_action("bicep_curl", bicep_curl_angle, BICEP_CURL_ANGLE_UP_THRESHOLD, BICEP_CURL_ANGLE_DOWN_THRESHOLD)

# GUI Setup
parent = tk.Tk()
frame = tk.Frame(parent)
frame.pack()

text_disp = tk.Button(frame, text="bicep_curl", command=write_text)
text_disp.pack(side=tk.LEFT)

exit_button = tk.Button(frame, text="Exit", fg="green", command=parent.quit)
exit_button.pack(side=tk.RIGHT)

left_hip_button = tk.Button(frame, text="lateral_raise", command=left_hip_action)
left_hip_button.pack(side=tk.LEFT)

parent.mainloop()