import gc
import ast
import tqdm
import time
import h5py
import glob
import json
import torch
import pandas as pd
import numpy as np
from collections import Counter
import torch.utils.data as torch_data
from torch.utils.data import Dataset
import logging
import random

from augmentations import augmentations

import cv2

def get_data_from_h5(path):
    hf = h5py.File(path, 'r')
    return hf

####################################################################
# Function that helps to see keypoints in an image
####################################################################
def prepare_keypoints_image(keypoints,tag):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}

    # DRAW POINTS
    img = np.zeros((256, 256, 3), np.uint8)

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)

        cv2.circle(img, (cor_x, cor_y), 1, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)

    cv2.imwrite(f'foo_{tag}.jpg', img)

##########################################################
# Process used to normalize the pose
##########################################################
def normalize_pose(data, body_dict):

    sequence_size = data.shape[0]
    valid_sequence = True

    last_starting_point, last_ending_point = None, None

    for sequence_index in range(sequence_size):

        # Prevent from even starting the analysis if some necessary elements are not present
        if (data[sequence_index][body_dict['pose_left_shoulder']][0] == 0.0 or data[sequence_index][body_dict['pose_right_shoulder']][0] == 0.0):
            if not last_starting_point:
                valid_sequence = False
                continue

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

            if data[sequence_index][body_dict['pose_left_shoulder']][0] != 0 and data[sequence_index][body_dict['pose_right_shoulder']][0] != 0:
                
                left_shoulder = data[sequence_index][body_dict['pose_left_shoulder']]
                right_shoulder = data[sequence_index][body_dict['pose_right_shoulder']]

                shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                                       (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)

                mid_distance = (0.5,0.5)#(left_shoulder - right_shoulder)/2
                head_metric = shoulder_distance/2
            '''
            # use it if you have the neck keypoint
            else:
                neck = (data["neck_X"][sequence_index], data["neck_Y"][sequence_index])
                nose = (data["nose_X"][sequence_index], data["nose_Y"][sequence_index])
                neck_nose_distance = ((((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)) ** 0.5)
                head_metric = neck_nose_distance
            '''
            # Set the starting and ending point of the normalization bounding box
            starting_point = [mid_distance[0] - 3 * head_metric, data[sequence_index][body_dict['pose_right_eye']][1] - (head_metric / 2)]
            ending_point = [mid_distance[0] + 3 * head_metric, mid_distance[1] + 4.5 * head_metric]

            last_starting_point, last_ending_point = starting_point, ending_point

        # Normalize individual landmarks and save the results
        for pos, kp in enumerate(data[sequence_index]):
            
            # Prevent from trying to normalize incorrectly captured points
            if data[sequence_index][pos][0] == 0:
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                    starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - ending_point[1]) / (starting_point[1] -
                                                                                    ending_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = 1 - normalized_y
            
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
    for sequence_index in range(sequence_size):

        # Retrieve all of the X and Y values of the current frame
        landmarks_x_values = data[sequence_index][:, 0]
        landmarks_y_values = data[sequence_index][:, 1]

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
        for pos, kp in enumerate(data[sequence_index]):

            # Prevent from trying to normalize incorrectly captured points
            if data[sequence_index][pos][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (
                    starting_point[1] - ending_point[1]) == 0:
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                    starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - starting_point[1]) / (ending_point[1] -
                                                                                    starting_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = normalized_y

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

    prepare_keypoints_image(data[2][0][leftHand+rightHand+pose,:],"before")

    for index_video in range(len(data)):
        data[index_video][:,pose,:] = normalize_pose(data[index_video][:,pose,:], body_section_dict)
        #data[index_video][:,face,:] = normalize_hand(data[index_video][:,face,:], body_section_dict)
        data[index_video][:,leftHand,:] = normalize_hand(data[index_video][:,leftHand,:], body_section_dict)
        data[index_video][:,rightHand,:] = normalize_hand(data[index_video][:,rightHand,:], body_section_dict)

    prepare_keypoints_image(data[2][0][leftHand+rightHand+pose,:],"after")

    kp_bp_index = {'pose':pose,
                   'left_hand':leftHand,
                   'rigth_hand':rightHand}

    return data, kp_bp_index, body_section_dict


def get_dataset_from_hdf5(path,keypoints_model,landmarks_ref,keypoints_number,threshold_frecuency_labels=10,list_labels_banned=[],dict_labels_dataset=None,
                         inv_dict_labels_dataset=None):
    print('path                       :',path)
    print('keypoints_model            :',keypoints_model)
    print('landmarks_ref              :',landmarks_ref)
    print('threshold_frecuency_labels :',threshold_frecuency_labels)
    print('list_labels_banned         :',list_labels_banned)
    
    # Prepare the data to process the dataset

    index_array_column = None #'mp_indexInArray', 'wp_indexInArray','op_indexInArray'

    print('Use keypoint model : ',keypoints_model) 
    if keypoints_model == 'openpose':
        index_array_column  = 'op_indexInArray'
    if keypoints_model == 'mediapipe':
        index_array_column  = 'mp_indexInArray'
    if keypoints_model == 'wholepose':
        index_array_column  = 'wp_indexInArray'
    print('use column for index keypoint :',index_array_column)

    assert not index_array_column is None

    # all the data from landmarks_ref
    df_keypoints = pd.read_csv(landmarks_ref, skiprows=1)

    # 29, 54 or 71 points
    if keypoints_number == 29:
        df_keypoints = df_keypoints[(df_keypoints['Selected 29']=='x' )& (df_keypoints['Key']!='wrist')]
    elif keypoints_number == 71:
        df_keypoints = df_keypoints[(df_keypoints['Selected 71']=='x' )& (df_keypoints['Key']!='wrist')]
    else:
        df_keypoints = df_keypoints[(df_keypoints['Selected 54']=='x')]

    logging.info(" using keypoints_number: "+str(keypoints_number))

    idx_keypoints = sorted(df_keypoints[index_array_column].astype(int).values)
    name_keypoints = df_keypoints['Key'].values
    section_keypoints = (df_keypoints['Section']+'_'+df_keypoints['Key']).values

    print('section_keypoints : ',len(section_keypoints),' -- uniques: ',len(set(section_keypoints)))
    print('name_keypoints    : ',len(name_keypoints),' -- uniques: ',len(set(name_keypoints)))
    print('idx_keypoints     : ',len(idx_keypoints),' -- uniques: ',len(set(idx_keypoints)))
    print('')
    print('section_keypoints used:')
    print(section_keypoints)

    # process the dataset (start)

    print('Reading dataset .. ')
    data = get_data_from_h5(path)
    #torch.Size([5, 71, 2])

    print('Total size dataset : ',len(data.keys()))
    print('Keys in dataset:', data.keys())
    video_dataset  = []
    labels_dataset = []

    video_name_dataset = []
    false_seq_dataset = []
    percentage_dataset = []
    max_consec_dataset = []
    time.sleep(2)
    for index in tqdm.tqdm(list(data.keys())):
        data_video = np.array(data[index]['data'])
        data_label = np.array(data[index]['label']).item().decode('utf-8')
        # F x C x K  (frames, coords, keypoitns)
        n_frames, n_axis, n_keypoints = data_video.shape

        data_video = np.transpose(data_video, (0,2,1)) #transpose to n_frames, n_keypoints, n_axis 
        if index=='0':
            print('original size video : ',data_video.shape,'-- label : ',data_label)
            print('filtering by keypoints idx .. ')
        data_video = data_video[:,idx_keypoints,:]

        if index=='0':
            print('filtered size video : ',data_video.shape,'-- label : ',data_label)

        data_video_name = np.array(data[index]['video_name']).item().decode('utf-8')
        #data_false_seq = np.array(data[index]['false_seq'])
        #data_percentage_groups = np.array(data[index]['percentage_group'])
        #data_max_consec = np.array(data[index]['max_percentage'])



        video_dataset.append(data_video)
        labels_dataset.append(data_label)
        video_name_dataset.append(data_video_name.encode('utf-8'))
        #false_seq_dataset.append(data_false_seq)
        #percentage_dataset.append(data_percentage_groups)
        #max_consec_dataset.append(data_max_consec)
        # # Get additional video attributes
        # videoname = np.array(data[index]['video_name']).item().decode('utf-8')
        # false_seq = np.array(data[index]['false_seq']).item()
        # percentage_groups = np.array(data[index]['percentage_group']).item()
        # max_consec = np.array(data[index]['max_percentage']).item()
    #     print("videoname:",videoname,"type:",type(videoname))                
    #     print("false_seq:",false_seq,"type:",type(false_seq))
    #     print("percentage_groups:",percentage_groups,"type:",type(percentage_groups))
    #     print("max_consec:",max_consec,"type:",type(max_consec))

    #     video_info.append((
    #         videoname,
    #         false_seq,
    #         percentage_groups,
    #         max_consec
    #     ))

    # print("video info shape:",len(video_info))

    del data
    gc.collect()
    
    if dict_labels_dataset is None:
        dict_labels_dataset = {}
        inv_dict_labels_dataset = {}

        for index,label in enumerate(sorted(set(labels_dataset))):
            dict_labels_dataset[label] = index
            inv_dict_labels_dataset[index] = label
    
    json_data = json.dumps(inv_dict_labels_dataset, indent=4)
    json_file_path = "meaning.json"
    with open(json_file_path, "w") as jsonfile:
        jsonfile.write(json_data)
    assert 1 == 2
    
    print('sorted(set(labels_dataset))  : ',sorted(set(labels_dataset)))
    print('dict_labels_dataset      :',dict_labels_dataset)
    print('inv_dict_labels_dataset  :',inv_dict_labels_dataset)
    encoded_dataset = [dict_labels_dataset[label] for label in labels_dataset]
    print('encoded_dataset:',len(encoded_dataset))

    print('label encoding completed!')

    print('total unique labels : ',len(set(labels_dataset)))
    print('Reading dataset completed!')

    return video_dataset, video_name_dataset, labels_dataset, encoded_dataset, dict_labels_dataset, inv_dict_labels_dataset, df_keypoints['Section'], section_keypoints

class LSP_Dataset(Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]  # type: ignore
    labels: [np.ndarray]  # type: ignore

    def __init__(self, dataset_filename: str,keypoints_model:str,  transform=None, have_aumentation=True,
                 augmentations_prob=0.5, normalize=False,landmarks_ref= 'Mapeo landmarks librerias.csv',
                dict_labels_dataset=None,inv_dict_labels_dataset=None, keypoints_number = 54):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """
        print("*"*20)
        print("*"*20)
        print("*"*20)
        print('Use keypoint model : ',keypoints_model) 
        logging.info('Use keypoint model : '+str(keypoints_model))

        self.list_labels_banned = []

        if  'AEC' in  dataset_filename:
            self.list_labels_banned += []

        if  'PUCP' in  dataset_filename:
            self.list_labels_banned += []
            self.list_labels_banned += []

        if  'WLASL' in  dataset_filename:
            self.list_labels_banned += []

        print('self.list_labels_banned',self.list_labels_banned)
        logging.info('self.list_labels_banned '+str(self.list_labels_banned))

        video_dataset, video_name_dataset, labels_dataset, encoded_dataset, dict_labels_dataset, inv_dict_labels_dataset, body_section, body_part = get_dataset_from_hdf5(path=dataset_filename,
                                                                                                                                       keypoints_model=keypoints_model,
                                                                                                                                       landmarks_ref=landmarks_ref,
                                                                                                                                       keypoints_number = keypoints_number,
                                                                                                                                       threshold_frecuency_labels =0,
                                                                                                                                       list_labels_banned =self.list_labels_banned,
                                                                                                                                       dict_labels_dataset=dict_labels_dataset,
                                                                                                                                       inv_dict_labels_dataset=inv_dict_labels_dataset)
        # HAND AND POSE NORMALIZATION
        video_dataset, keypoint_body_part_index, body_section_dict = normalize_pose_hands_function(video_dataset, body_section, body_part)

        self.data = video_dataset
        self.video_name = video_name_dataset
        #self.false_seq = false_seq_dataset
        #self.percentage = percentage_dataset
        #self.max_consec = max_consec_dataset
        self.labels = encoded_dataset
        self.label_freq = Counter(self.labels)
        #self.targets = list(encoded_dataset)
        self.text_labels = list(labels_dataset)
        self.transform = transform
        self.dict_labels_dataset = dict_labels_dataset
        self.inv_dict_labels_dataset = inv_dict_labels_dataset
        
        self.have_aumentation = have_aumentation
        print(keypoint_body_part_index, body_section_dict)
        self.augmentation = augmentations.augmentation(keypoint_body_part_index, body_section_dict)
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize


    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """
        depth_map = torch.from_numpy(np.copy(self.data[idx]))


        # Apply potential augmentations
        if self.have_aumentation and random.random() < self.augmentations_prob:

            selected_aug = random.randrange(4)

            if selected_aug == 0:
                depth_map = self.augmentation.augment_rotate(depth_map, angle_range=(-13, 13))

            if selected_aug == 1:
                depth_map = self.augmentation.augment_shear(depth_map, "perspective", squeeze_ratio=(0, 0.1))

            if selected_aug == 2:
                depth_map = self.augmentation.augment_shear(depth_map, "squeeze", squeeze_ratio=(0, 0.15))

            if selected_aug == 3:
                depth_map = self.augmentation.augment_arm_joint_rotate(depth_map, 0.3, angle_range=(-4, 4))





        video_name = self.video_name[idx].decode('utf-8')
        #false_seq = self.false_seq
        #percentage_group = self.percentage
        #max_consec = self.max_consec
        label = torch.Tensor([self.labels[idx]])
        depth_map = depth_map - 0.5
        if self.transform:
            depth_map = self.transform(depth_map)
        return depth_map, label, video_name #, false_seq, percentage_group, max_consec

    def __len__(self):
        return len(self.labels)

