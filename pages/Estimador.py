"""
Estimador.py

DESCRIPTION: generate the pose estimatios.

AUTHOR: Juan Ignacio Ribet
DATE: 08-Sep-2023
"""

import time
import pickle
import cv2
import json
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import math
import tempfile



def camera_relese():
    try:
        cap.release()
    except:
        pass

def inFrame(lst):
    '''
    Check if specific landmarks in a list of facial landmarks are visible.

    This function checks the visibility confidence of specific body landmarks (landmark indices
    15, 16, 27, and 28) and returns True if at least one of the landmarks on each side of the face
    is visible with a confidence greater than 0.6.

    Parameters:
        lst (list): A list containing facial landmarks, where each landmark is represented as an
                    object with attributes like 'visibility'.

    Returns:
        bool: True if at least one landmark on each side of the face is visible with confidence
              greater than 0.6, False otherwise.
    '''
    if ((lst[28].visibility > 0.6 or lst[27].visibility > 0.6)
            and (lst[15].visibility > 0.6 or lst[16].visibility > 0.6)):
        return True
    return False


def calculate_angle_coord(p_cood_list):
    """
    Calculate the angle formed by three coordinates in a 2D plane.

    Parameters:
        p_cood_list (list): A list containing three 2D coordinate points as numpy arrays.

    Returns:
        float: The angle in degrees between the lines connecting the first and second points
               and the second and third points. The angle is always in the range [0, 180].
    
    first_point = p_cood_list[0][*]
    mid_point = p_cood_list[1][*]
    last_point = p_cood_list[2][*]
    """
    radians = math.atan2(p_cood_list[2][1]-p_cood_list[1][1], p_cood_list[2][0]-p_cood_list[1][0]) - \
        math.atan2(p_cood_list[0][1]-p_cood_list[1][1], p_cood_list[0][0]-p_cood_list[1][0])
    angle = abs(radians*180.0/math.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)


if __name__== '__main__':

    im = Image.open('pages/Data/loto.png')

    st.set_page_config(
        page_title="Detector de posturas de Yoga",
        page_icon=im,
        layout="wide")

    st.title("Yoga Pose Detector")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria_expanded="true"] > dic:first-child{
            width:350x
        }

        [data-testid="stSidebar"][aria_expanded="false"] > dic:first-child{
            width:350x
            margin-left: -350x
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    """
    Presionar el botón "Iniciar" para empezar la estimación de postura en el entrenamiento.
    Al finalizar presionar el botón "Fin"

    Puede procesar un video propio o ver el video de ejemplo presionando el botón "video demostración".
    Tildando "Procesado rápido" el sistema procesa solo 15 cuadros por segundo (puede modificar los resultados).
    """


    col1, col2 = st.columns([1, 5])
    with col1:
        run = st.button('Iniciar')
    with col2:
        stop = st.button("Fin")

    frame_placeholder = st.empty()
    placeholder = st.empty()
    placeholder2 = st.empty()

    save = st.sidebar.checkbox('Guardar video procesado')

    st.sidebar.write("---")

    uploaded_file = st.sidebar.file_uploader("Carga video para procesar")
    if uploaded_file is not None:
        procesar_video = st.sidebar.button(':green[Procesar video]')
        if procesar_video:
            run = True

    fast_process = st.sidebar.checkbox('Procesado rapido')

    demo_video = st.sidebar.button(':blue[video demostración]')
    if demo_video:
        run = True
    
    st.sidebar.write("---")

    # Select the camera
    camera = st.sidebar.number_input(
        'seleccionar camara', value=0, max_value=2, on_change=camera_relese)
    
    with open('pages\Data\models\model_svm.pkl', 'rb') as f:
        model = pickle.load(f)

    mp_pose = mp.solutions.pose  # Mediapipe Solutions
    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers

    landmarks = []
    for val in range(0, 33):
        landmarks += ['x{}'.format(val), 
                    'y{}'.format(val),
                    'z{}'.format(val), 
                    'v{}'.format(val)]


    # Define the angles to messure
    dic = {'downdog': {(23, 11, 13): 169,
                    (24, 12, 14): 169,
                    (11, 13, 15): 166,
                    (12, 14, 16): 166,
                    (11, 23, 25): 56,
                    (12, 24, 26): 56,
                    (23, 25, 27): 174,
                    (24, 26, 28): 174,},
            'goddess': {(11, 23, 25): 108,
                        (12, 24, 26): 108, 
                        (23, 25, 27): 114, 
                        (24, 26, 28): 114,},
            'tree': {(11, 23, 25): 120,
                    (12, 24, 26): 174, 
                    (23, 25, 27): 56,
                    (24, 26, 28): 176},
            'tree_inv': {(11, 23, 25): 174,
                        (12, 24, 26): 120, 
                        (23, 25, 27): 176,
                        (24, 26, 28): 56},
            'warrior': {(13, 11, 23): 95,
                        (14, 12, 24): 97,
                        (11, 13, 15): 172,
                        (16, 14, 12): 171,
                        (11, 23, 25): 100,
                        (12, 24, 26): 135,
                        (23, 25, 27): 110,
                        (24, 26, 28): 170},
            'warrior_inv': {(13, 11, 23): 95,
                            (14, 12, 24): 97,
                            (11, 13, 15): 172,
                            (16, 14, 12): 171,
                            (11, 23, 25): 135,
                            (12, 24, 26): 100,
                            (23, 25, 27): 170,
                            (24, 26, 28): 110}
            }


    # Images to display on screen
    dic_images = {'downdog': 'pages/Data/images_display/video/downdog.png',
                    'warrior': 'pages/Data/images_display/video/warrior.png',
                    'warrior_inv': 'pages/Data/images_display/video/warrior_inv.png',
                    'goddess': 'pages/Data/images_display/video/goddess.png',
                    'tree': 'pages/Data/images_display/video/Tree.png',
                    'tree_inv': 'pages/Data/images_display/video/tree_inv.png'
                    }

    if run:
        if demo_video:
            cap = cv2.VideoCapture("pages\Data\\sample_video.mp4")
            #cap.set(cv2.CAP_PROP_FPS,5)
            fps_video = cap.get(cv2.CAP_PROP_FPS)

        elif uploaded_file is not None:
            if procesar_video:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
                #cap.set(cv2.CAP_PROP_FPS,5)
                fps_video = int(cap.get(cv2.CAP_PROP_FPS))
        
        else:
            cap = cv2.VideoCapture(camera)
            pass
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        result = cv2.VideoWriter('pages/Data/video.mp4', 
                                cv2.VideoWriter_fourcc(*'VIDX'),
                                20, 
                                (frame_width, frame_height))




        if not cap.isOpened():
            st.sidebar.text(f'La camara {camera} no está disponible')
            cap = cv2.VideoCapture(0)

        # Iniciate variables
        
        frame_number = 0
        body_language = 0
        body_language_time = 0
        pose_time = 0
        report = []
        report_time = []
        prev_frame_time = 0
        new_frame_time = 0

        start_time = time.time()

        with mp_pose.Pose(model_complexity=1, 
                        smooth_landmarks = True, 
                        min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5) as pose:

            while cap.isOpened():
                success, frame = cap.read()
                if stop:
                    break
                if not success:
                    st.sidebar.write('the video capture end')
                    break
                if frame is None:
                    break

                if (demo_video or (uploaded_file is not None)) and fast_process:
                    if frame_number % int((fps_video/15)) == 0:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                    else:
                        frame_number += 1
                        continue
                else:
                    # Recolor Feed
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                
                # Mido FPS
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time    
                fps = int(fps)
                # Make Detections
                results = pose.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Export coordinates
                try:
                    circle_coord = ((frame_width-100), 40)

                    if results.pose_landmarks and inFrame(results.pose_landmarks.landmark):

                        # Draw the landmarks connections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                                mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                        thickness=1,
                                                                        circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(219, 170, 117),
                                                                        thickness=1,
                                                                        circle_radius=2)
                                                )

                        # Extract Pose landmarks
                        poses = results.pose_landmarks.landmark
                        row = list(np.array(
                            [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in poses]).flatten())

                        # Make Detections every 3 seconds
                        time_laps = 3
                        current_time = int(time.time()-start_time)
                        if (body_language == 0) or (current_time % time_laps == 0):
                            X = pd.DataFrame([row], columns=landmarks)
                            body_language_class = model.predict(X)[0]
                            body_language_prob = model.predict_proba(X)[0]


                        if body_language_prob[np.argmax(body_language_prob)] < 0.50:
                            body_language = 0

                        # Draw the status box
                        cv2.rectangle(image,
                                    (0, 0),
                                    (600, 60),
                                    (0, 0, 0),
                                    -1)

                        # Display pose detected
                        cv2.putText(image,
                                    'POSE',
                                    (195, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (250, 250, 250),
                                    1,
                                    cv2.LINE_AA)

                        # Check if the pose detection probability is greater than 80%
                        if body_language_prob[np.argmax(body_language_prob)] >= 0.80:
                            
                            body_language = body_language_class

                            # record the time for each posture.
                            if (body_language != 0) & (body_language != body_language_time):
                                finish_pose_time = int(time.time()-pose_time)
                                pose_time = time.time()
                                if body_language_time != 0:
                                    report_time.append([body_language_time,finish_pose_time])

                                body_language_time = body_language
                            
                            
                            cv2.putText(image,
                                        body_language_class.split(' ')[0],
                                        (190, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (255, 255, 255),
                                        2,
                                        cv2.LINE_AA)
                            cv2.circle(image, 
                                    circle_coord, 
                                    40, 
                                    (0, 255, 0), 
                                    -1)
                        else:
                            cv2.circle(image,
                                    circle_coord,
                                    40,
                                    (0, 0, 255),
                                    -1)

                        # Display Probability
                        cv2.putText(image,
                                    'PROB',
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (250, 250, 250),
                                    1,
                                    cv2.LINE_AA)
                        cv2.putText(image,
                                    f'{body_language_prob[np.argmax(body_language_prob)]:.0%}',
                                    (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA)

                        if body_language:
                            # The poses 'warrior' and 'tree' are not symmetrical. The next lines check  
                            # the angle in P26 to check whether the pose detected is rigth or left
                            # and select the angles to measure in correspondence.
                            dato_pose = body_language_class
                            if body_language_class in ['warrior', 'tree']:
                                angle_1 = 0
                                try:
                                    angle_1 = int(calculate_angle_coord([(
                                        results.pose_landmarks.landmark[24].x,
                                        results.pose_landmarks.landmark[24].y),
                                        (
                                        results.pose_landmarks.landmark[26].x,
                                        results.pose_landmarks.landmark[26].y),
                                        (
                                        results.pose_landmarks.landmark[28].x,
                                        results.pose_landmarks.landmark[28].y)]))

                                except:
                                    pass
                                if angle_1 < 150:
                                    class_inv = body_language_class + '_inv'
                                    p_dic = dic.get(class_inv)
                                    dato_pose = class_inv
                                else:
                                    p_dic = dic.get(body_language_class)

                            else:
                                p_dic = dic.get(body_language_class)
                            
                            
                            q_ang = 0 # inicilice the counting of the angles measured
                            q_ang_ok = 0 # inicilice the counting of the angles measured ok

                            # Extract the angles points to measure from angle dict.
                            for i in range(len(p_dic)):
                                p_cood_list = []
                                midle_point = list(p_dic.keys())[i][1]
                                for p in list(p_dic.keys())[i]:
                                    if results.pose_landmarks.landmark[p].visibility > 0.5:
                                        
                                        p_corrd = (
                                            results.pose_landmarks.landmark[p].x,
                                            results.pose_landmarks.landmark[p].y)
                                        p_cood_list.append(p_corrd)
                                    else:
                                        break

                                try:
                                    # Meassure the angles
                                    angle = int(calculate_angle_coord(p_cood_list))
                                    angle_ok = int(list(p_dic.values())[i])
                                    q_ang += 1
                                    report.append([dato_pose,
                                                list(p_dic.keys())[i],
                                                angle_ok,
                                                angle])

                                    # Print the angles into the image. Green if it is between the 
                                    # tolerance and red if it is not.
                                    tolerance = 15
                                    if angle in range(angle_ok-tolerance, angle_ok+tolerance):
                                        q_ang_ok += 1
                                        text_color = (0, 255, 0)
                                    else:
                                        text_color = (0, 0, 255)

                                    cv2.putText(image,
                                                str(angle),
                                                (int((results.pose_landmarks.landmark[midle_point].x) \
                                                    *image.shape[1]),
                                                int((results.pose_landmarks.landmark[midle_point].y) \
                                                    *image.shape[0])),
                                                cv2.FONT_HERSHEY_PLAIN,
                                                2,
                                                text_color,
                                                2)
                                    cv2.putText(image,
                                                f'({angle_ok})',
                                                (int((results.pose_landmarks.landmark[midle_point].x) \
                                                    *image.shape[1]),
                                                int((results.pose_landmarks.landmark[midle_point].y) \
                                                    *image.shape[0])+25),
                                                cv2.FONT_HERSHEY_PLAIN,
                                                1,
                                                (66, 245, 236),
                                                1)

                                except:
                                    pass

                            # Insert the example picture into the image
                            img_path = dic_images.get(dato_pose)
                            img = cv2.imread(img_path)
                            h = img.shape[0]
                            w = img.shape[1]
                            image[image.shape[0]-(h+10):image.shape[0]-10,
                                    image.shape[1]-(w+10):image.shape[1]-10] = img
                            
                            # if any angle is out of range a red led turns on
                            if q_ang_ok != q_ang:
                                cv2.circle(image,
                                circle_coord,
                                20,
                                (0, 0, 250),
                                -1)

                    # Print if the body is not fully visible                    
                    else:
                        cv2.putText(image,
                                    "Make your Full",
                                    (50, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (66, 245, 236),
                                    3)
                        cv2.putText(image,
                                    "body visible",
                                    (50, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (66, 245, 236),
                                    3)
                        cv2.circle(image,
                                circle_coord,
                                40,
                                (0, 0, 255),
                                -1)

                except:
                    pass
    
                cv2.putText(image, 
                            f'FPS: {str(fps)}', 
                            (15, frame_height-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1 , 
                            (250, 250, 250),
                            2)
                cv2.putText(image, f'Frame {frame_number}', (15, frame_height-20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                #cv2.putText(image, f'fps total {fps1}', (200, frame_height-20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                frame_number += 1
                if save:
                    result.write(image)                    
                frame_placeholder.image(image, channels='BGR')
                                
                # Show the time 
                placeholder.text(f'time: {int(time.time()-start_time)}')          

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # Save the results
        report_df = pd.DataFrame(report, columns=['pose',
                                                    'punto',
                                                    'ang optimo',
                                                    'ang medido medio'])
        report_df.to_csv(
            'pages/Data/report.csv', index=False)
        
        # Clean assistant file
        dict_filename = 'pages/Data/assit_resp.json'
        dic_assist_resp = {}
        save_dict_to_file(dic_assist_resp, dict_filename)

        
        finish_pose_time = int(time.time()-pose_time)
        
        # Save last posture time
        pose_time = time.time()
        report_time.append([body_language_time,finish_pose_time])
        report_time_df = pd.DataFrame(report_time, columns=['pose','time'])
        report_time_df.to_csv(
                'pages/Data/report_time.csv', index=False)  

        result.release()            
        cap.release()
        cv2.destroyAllWindows()
