import mediapipe as mp
import pandas as pd
import numpy as np
import cv2 
import torch
import json
import sys
import os

from collections import OrderedDict

from spoter.spoter_model import SPOTER

print('HIDDEN_DIM and NUM_CLASSES - READY')

def get_mp_keys(points):
    tar = np.array(points.mp_pos)-1
    return list(tar)

def get_op_keys(points):
    tar = np.array(points.op_pos)-1
    return list(tar)

def get_wp_keys(points):
    tar = np.array(points.wb_pos)-1
    return list(tar)

def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    
    if isinstance(config_file, str):
        with open(config_file, 'r') as json_file:
            return json.load(json_file)
    else:
        return config_file

def configure_model(config_file, use_wandb):

    config_file = parse_configuration(config_file)

    config = dict(
        #hidden_dim = config_file["hparams"]["hidden_dim"],
        num_classes = config_file["hparams"]["num_classes"],
        epochs = config_file["hparams"]["epochs"],
        num_backups = config_file["hparams"]["num_backups"],
        keypoints_model = config_file["hparams"]["keypoints_model"],
        lr = config_file["hparams"]["lr"],
        keypoints_number = config_file["hparams"]["keypoints_number"],

        nhead = config_file["hparams"]["nhead"],
        num_encoder_layers = config_file["hparams"]["num_encoder_layers"],
        num_decoder_layers = config_file["hparams"]["num_decoder_layers"],
        dim_feedforward = config_file["hparams"]["dim_feedforward"],

        experimental_train_split = config_file["hparams"]["experimental_train_split"],
        validation_set = config_file["hparams"]["validation_set"],
        validation_set_size = config_file["hparams"]["validation_set_size"],
        log_freq = config_file["hparams"]["log_freq"],
        save_checkpoints = config_file["hparams"]["save_checkpoints"],
        scheduler_factor = config_file["hparams"]["scheduler_factor"],
        scheduler_patience = config_file["hparams"]["scheduler_patience"],
        gaussian_mean = config_file["hparams"]["gaussian_mean"],
        gaussian_std = config_file["hparams"]["gaussian_std"],
        plot_stats = config_file["hparams"]["plot_stats"],
        plot_lr = config_file["hparams"]["plot_lr"],

        #training_set_path = config_file["data"]["training_set_path"],
        #validation_set_path = config_file["data"]["validation_set_path"],
        #testing_set_path = config_file["data"]["testing_set_path"],

        n_seed = config_file["seed"],
        device = config_file["device"],
        dataset_path = config_file["dataset_path"],
        weights_trained = config_file["weights_trained"],
        save_weights_path = config_file["save_weights_path"],
        dataset = config_file["dataset"]
    )
    return config

def frame_process(holistic, frame):

    imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = imageBGR.shape

    holisResults = holistic.process(imageBGR)

    kpDict = {}

    #POSE

    if holisResults.pose_landmarks:
        pose = [ [point.x, point.y] for point in holisResults.pose_landmarks.landmark]
    else:
        pose = [ [0.0, 0.0] for point in range(0, 33)]
    pose = np.asarray(pose)

    # HANDS

    # Left hand
    if(holisResults.left_hand_landmarks):
        left_hand = [ [point.x, point.y] for point in holisResults.left_hand_landmarks.landmark]
    else:
        left_hand = [ [pose[15][0], pose[15][1]] for point in range(0, 21)]
    left_hand = np.asarray(left_hand)

    # Right hand
    if(holisResults.right_hand_landmarks):
        right_hand = [ [point.x, point.y] for point in holisResults.right_hand_landmarks.landmark]
    else:
        right_hand = [[pose[16][0], pose[16][1]] for point in range(0, 21)]
    right_hand = np.asarray(right_hand)

    # Face mesh

    if(holisResults.face_landmarks):

        face = [ [point.x, point.y] for point in holisResults.face_landmarks.landmark]

    else:
        face = [[0.0, 0.0] for point in range(0, 468)]
    face = np.asarray(face)

    neck = (pose[11] + pose[12]) / 2

    newFormat = []

    newFormat.append(pose)
    newFormat.append(face)
    newFormat.append(left_hand)
    newFormat.append(right_hand)
    newFormat.append([neck])

    x = np.asarray([item[0] for sublist in newFormat for item in sublist])
    y = np.asarray([item[1] for sublist in newFormat for item in sublist])

    data = np.asarray([x,y])

    return data


def preprocess_video(data):
    model_key_getter = {'mediapipe': get_mp_keys,
                    'openpose': get_op_keys,
                    'wholepose': get_wp_keys}

    model_key_getter = model_key_getter['mediapipe']

    # 71 or 29 points
    num_joints = 54
    
    points = pd.read_csv(f"./points_{num_joints}.csv")

    print("LOADING DATA FROM S3")
    bucket_name = 'user-video-test'
    file_path = data

    print(file_path, file_path)

    print("DATA LOADED")
    input_path = f'./{data}'

    path = input_path
    print(path)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    cap = cv2.VideoCapture(path)
    #length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))

    print(path)

    output_list = []
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break 

        pred = frame_process(holistic, img)

        selected_joints = model_key_getter(points)
        
        pred = pred[:,selected_joints]

        output_list.append(pred)

        success, img = cap.read()

    holistic.close()

    result = np.asarray(output_list)

    return result

##########################################################
# Process used to normalize the pose
##########################################################
def normalize_pose(data, body_dict):

    valid_sequence = True

    last_starting_point, last_ending_point = None, None

    # Prevent from even starting the analysis if some necessary elements are not present
    if (data[body_dict['pose_left_shoulder']][0] == 0.0 or data[body_dict['pose_right_shoulder']][0] == 0.0):
        if not last_starting_point:
            valid_sequence = False
        else:
            starting_point, ending_point = last_starting_point, last_ending_point

    else:

        # NOTE:
        #
        # While in the paper, it is written that the head metric is calculated by halving the shoulder distance,
        # this is meant for the distance between the very ends of one's shoulder, as literature studying body
        # metrics and ratios generally states. The Vision Pose Estimation API, however, seems to be predicting
        # rather the center of one's shoulder. Based on our experiments and manual reviews of the data, employing
        # this as just the plain shoulder distance seems to be more corresponding to the desired metric.
        #
        # Please, review this if using other third-party pose estimation libraries.

        if data[body_dict['pose_left_shoulder']][0] != 0 and data[body_dict['pose_right_shoulder']][0] != 0:
            
            left_shoulder = data[body_dict['pose_left_shoulder']]
            right_shoulder = data[body_dict['pose_right_shoulder']]

            shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                                    (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)

            mid_distance = (0.5,0.5)#(left_shoulder - right_shoulder)/2
            head_metric = shoulder_distance/2
        '''
        # use it if you have the neck keypoint
        else:
            neck = (data["neck_X"], data["neck_Y"])
            nose = (data["nose_X"], data["nose_Y"])
            neck_nose_distance = ((((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)) ** 0.5)
            head_metric = neck_nose_distance
        '''
        # Set the starting and ending point of the normalization bounding box
        starting_point = [mid_distance[0] - 3 * head_metric, data[body_dict['pose_right_eye']][1] - (head_metric / 2)]
        ending_point = [mid_distance[0] + 3 * head_metric, mid_distance[1] + 4.5 * head_metric]

        last_starting_point, last_ending_point = starting_point, ending_point

    # Normalize individual landmarks and save the results
    for pos, kp in enumerate(data):
        
        # Prevent from trying to normalize incorrectly captured points
        if data[pos][0] == 0:
            continue

        normalized_x = (data[pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                starting_point[0])
        normalized_y = (data[pos][1] - ending_point[1]) / (starting_point[1] -
                                                                                ending_point[1])

        data[pos][0] = normalized_x
        data[pos][1] = 1 - normalized_y
            
    return data
################################################
# Function that normalize the hands (but also the face)
################################################
def normalize_hand(data, body_section_dict):
    """
    Normalizes the skeletal data for a given sequence of frames with signer's hand pose data. The normalization follows
    the definition from our paper.
    :param data: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                that particular joints coordinates
    :return: Dictionary with normalized skeletal data (following the same schema as input data)
    """

    sequence_size = data.shape[0]
    
    # Treat each element of the sequence (analyzed frame) individually


    # Retrieve all of the X and Y values of the current frame
    landmarks_x_values = data[:, 0]
    landmarks_y_values = data[:, 1]

    # Prevent from even starting the analysis if some necessary elements are not present
    #if not landmarks_x_values or not landmarks_y_values:
    #    continue

    # Calculate the deltas
    width, height = max(landmarks_x_values) - min(landmarks_x_values), max(landmarks_y_values) - min(
        landmarks_y_values)
    if width > height:
        delta_x = 0.1 * width
        delta_y = delta_x + ((width - height) / 2)
    else:
        delta_y = 0.1 * height
        delta_x = delta_y + ((height - width) / 2)

    # Set the starting and ending point of the normalization bounding box
    starting_point = (min(landmarks_x_values) - delta_x, min(landmarks_y_values) - delta_y)
    ending_point = (max(landmarks_x_values) + delta_x, max(landmarks_y_values) + delta_y)

    # Normalize individual landmarks and save the results
    for pos, kp in enumerate(data):

        # Prevent from trying to normalize incorrectly captured points
        if data[pos][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (
                starting_point[1] - ending_point[1]) == 0:
            continue

        normalized_x = (data[pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                starting_point[0])
        normalized_y = (data[pos][1] - starting_point[1]) / (ending_point[1] -
                                                                                starting_point[1])

        data[pos][0] = normalized_x
        data[pos][1] = normalized_y

    return data

###################################################################################
# This function normalize the body and the hands separately
# body_section has the general body part name (ex: pose, face, leftHand, rightHand)
# body_part has the specific body part name (ex: pose_left_shoulder, face_right_mouth_down, etc)
###################################################################################
def normalize_pose_hands_function(data, body_section, body_part):

    pose = [pos for pos, body in enumerate(body_section) if body == 'pose' or body == 'face']
    face = [pos for pos, body in enumerate(body_section) if body == 'face']
    leftHand = [pos for pos, body in enumerate(body_section) if body == 'leftHand']
    rightHand = [pos for pos, body in enumerate(body_section) if body == 'rightHand']

    body_section_dict = {body:pos for pos, body in enumerate(body_part)}

    assert len(pose) > 0 and len(leftHand) > 0 and len(rightHand) > 0 #and len(face) > 0

    for index_video in range(len(data)):
        data[index_video][pose,:] = normalize_pose(data[index_video][pose,:], body_section_dict)
        #data[index_video][face,:] = normalize_hand(data[index_video][face,:], body_section_dict)
        data[index_video][leftHand,:] = normalize_hand(data[index_video][leftHand,:], body_section_dict)
        data[index_video][rightHand,:] = normalize_hand(data[index_video][rightHand,:], body_section_dict)

    return data

def preprocess_keypoints(data):

    # Spoter format
    df_keypoints = pd.read_csv("./Mapeo landmarks librerias.csv", skiprows=1)
    df_keypoints = df_keypoints[(df_keypoints['Selected 54']=='x')]
    body_part = (df_keypoints['Section']+'_'+df_keypoints['Key']).values
    body_section = df_keypoints['Section']

    data = np.moveaxis(data, 1, 2)

    data = normalize_pose_hands_function(data, body_section, body_part)

    #data = np.moveaxis(data, 1, 2)

    depth_map = torch.from_numpy(np.copy(data))
    depth_map = depth_map - 0.5

    return depth_map

def preproccess(data):
    print("PREPROCESS_VIDEO")
    data = preprocess_video(data)
    print("PREPROCESS_KEYPOINT")
    data = preprocess_keypoints(data)

    return data

def load_model(path):

    CONFIG_FILENAME = "./config.json"

    config = configure_model(CONFIG_FILENAME, False)
    print(f'cwd - {str(os.getcwd())}')
    print(f'cwd list - {str(os.listdir(os.getcwd()))}')
    print(str(config))
    print("SPOTER")
    print(F"PATH - {path +'/'+ 'model.pth'}")
    model = SPOTER(num_classes=config['num_classes'], hidden_dim=config['keypoints_number'])
    '''
    model = SPOTER(num_classes=, 
                        hidden_dim=,
                        dim_feedforward=config['dim_feedforward'],
                        num_encoder_layers=config['num_encoder_layers'],
                        num_decoder_layers=config['num_decoder_layers'],
                        nhead=config['nhead']
                        )
    '''
    device = torch.device('cpu')
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("LOADED MODEL")
    return model

#####################################################################################
# SAGEMAKER PYTORCH SERVE DEF
#
# It have to be necessary these four def:
#
# def input_fn(request_body, request_content_type)
# def model_fn(model_dir)
# def predict_fn(input_data, model)
# def output_fn(prediction, content_type)



########################
#  This function receive a Json call from AWS Lambda and retrieve the information (the video name)
#  Then preprocess the data using a process similarly than ConnectingPoints repository 
########################
def input_fn(request_body, request_content_type):
    print(f'initialize input_fn - {request_content_type}')
    #print(request_body)
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/json':
        #data_json = json.loads(request_body)
        data = request_body["uniqueName"]

        #return torch.load(BytesIO(request_body))
    else:
        # Handle other content-types here or raise an Exception
        raise ValueError("The message content type should be Json to process the information")
        # if the content type is not supported.
        pass
    
    data = preproccess(data)

    #data = torch.Tensor(data)
    #data = Variable(data.float().cpu(), requires_grad=False)

    return data #"input"#data.transpose(3,1).transpose(2,3)

########################
# This function load the pre-trained model for SLR
########################
def model_fn(model_dir): 
    print(f'model_fn - {model_dir}')
    model = load_model(model_dir)
    return model

########################
# This function use the two outputs from "model_fn" and "input_fn"
# and to do the inference
# Note: the model has to be set to CPU
########################
def predict_fn(input_data, model):
    print("predict_fn")
    device = torch.device('cpu')
    print("initially set to use CPU")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Now using CUDA")
        device = torch.device("cuda")

    device = "cpu" # HARDCODED TO CPU

    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_data.to(device)).expand(1, -1, -1)

    return output

########################
# this function use the result from "predict_fn".
# I manually check the contect_type value and it is "application/json" I believe that this is the same
# than the type asked when AWS Lambda funciton call the inference (but maybe is the default value).
#
# The prediction is process to get the top 5
# then, using "meaning.json" we retrieve the word in spanish from the label
# then the information is formated to be sent to AWS Lambda as a response
########################
def output_fn(prediction, content_type):
    print(f'output_fn - {content_type}')
    _, predict_label = torch.topk(prediction.data, 5)

    result = predict_label[0][0].numpy() 

    with open('./meaning.json') as f:
        meaning = json.load(f)

    result = [meaning[str(val)] for val in result]
    print(result)
    if content_type == "application/json":
        instances = []
        for row in result:
            instances.append({"gloss": row})

        json_output = {"instances": instances}
        response = json.dumps(json_output)

    else:
        response = json.dumps({'NAI':['please, use application/json']})

    return response

data = input_fn({"uniqueName":"fuerte_493.mp4"},"application/json")
model = model_fn('spoter-50Classes-68_5Top1acc_87Top5acc.pth')
pred = predict_fn(data, model)
print(output_fn(pred, "application/json"))