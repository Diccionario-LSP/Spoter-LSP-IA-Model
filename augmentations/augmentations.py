import math
import logging
import cv2
import random
import torch

import numpy as np




class augmentation():
    
    def __init__(self, body_type_identifiers, body_section_dict):
        super().__init__()
        print(body_type_identifiers.keys())
        self.body_section_dict = body_section_dict
        self.BODY_IDENTIFIERS = body_type_identifiers['pose']
        self.HAND_IDENTIFIERS = body_type_identifiers['left_hand'] + body_type_identifiers['rigth_hand']
        
        Left_hand_id = ['pose_chest_middle_up', 'pose_left_shoulder', 'pose_left_elbow', 'pose_left_wrist']
        right_hand_id = ['pose_chest_middle_up', 'pose_right_shoulder', 'pose_right_elbow', 'pose_right_wrist']

        self.ARM_IDENTIFIERS_ORDER = [[body_section_dict[_id] for _id in Left_hand_id ],
                                      [body_section_dict[_id] for _id in right_hand_id]]

        '''
        arms_identifiers = ['pose_chest_middle_up', 'pose_right_wrist', 'pose_left_wrist','pose_right_elbow','pose_left_elbow', 'pose_left_shoulder', 'pose_right_shoulder']
        self.ARM_IDENTIFIERS_ORDER = [body_section_dict[identifiers] for identifiers in arms_identifiers]
        '''

    def __random_pass(self, prob):
        return random.random() < prob


    def __numpy_to_dictionary(self, data_array: np.ndarray) -> dict:
        """
        Supplementary method converting a NumPy array of body landmark data into dictionaries. The array data must match the
        order of the BODY_IDENTIFIERS list.
        """

        output = {}

        for landmark_index, identifier in enumerate(self.BODY_IDENTIFIERS):
            output[identifier] = data_array[:, landmark_index].tolist()

        return output


    def __dictionary_to_numpy(self, landmarks_dict: dict) -> np.ndarray:
        """
        Supplementary method converting dictionaries of body landmark data into respective NumPy arrays. The resulting array
        will match the order of the BODY_IDENTIFIERS list.
        """

        output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS), 2))

        for landmark_index, identifier in enumerate(BODY_IDENTIFIERS):
            output[:, landmark_index, 0] = np.array(landmarks_dict[identifier])[:, 0]
            output[:, landmark_index, 1] = np.array(landmarks_dict[identifier])[:, 1]

        return output


    def __rotate(self, origin: tuple, point: tuple, angle: float):
        """
        Rotates a point counterclockwise by a given angle around a given origin.
        :param origin: Landmark in the (X, Y) format of the origin from which to count angle of rotation
        :param point: Landmark in the (X, Y) format to be rotated
        :param angle: Angle under which the point shall be rotated
        :return: New landmarks (coordinates)
        """

        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy


    def __preprocess_row_sign(self, sign: dict) -> (dict, dict):
        """
        Supplementary method splitting the single-dictionary skeletal data into two dictionaries of body and hand landmarks
        respectively.
        """

        #sign_eval = sign

        body_landmarks = sign[:,self.BODY_IDENTIFIERS,:]
        hand_landmarks = sign[:,self.HAND_IDENTIFIERS,:]
        '''
        if "nose_X" in sign_eval:
            body_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                            for identifier in BODY_IDENTIFIERS}
            hand_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                            for identifier in HAND_IDENTIFIERS}

        else:
            body_landmarks = {identifier: sign_eval[identifier] for identifier in BODY_IDENTIFIERS}
            hand_landmarks = {identifier: sign_eval[identifier] for identifier in HAND_IDENTIFIERS}
        '''
        return body_landmarks, hand_landmarks


    def __wrap_sign_into_row(self, body_identifiers: dict, hand_identifiers: dict) -> dict:
        """
        Supplementary method for merging body and hand data into a single dictionary.
        """

        #return {**body_identifiers, **hand_identifiers}
        body_landmarks = torch.tensor(body_identifiers)
        hand_landmarks = torch.tensor(hand_identifiers)

        # Concatenar los dos tensores a lo largo de la segunda dimensión
        tensor_concatenado = torch.cat([body_landmarks, hand_landmarks], dim=1)
        return tensor_concatenado


    def augment_rotate(self, sign: dict, angle_range: tuple) -> dict:
        """
        AUGMENTATION TECHNIQUE. All the joint coordinates in each frame are rotated by a random angle up to 13 degrees with
        the center of rotation lying in the center of the frame, which is equal to [0.5; 0.5].
        :param sign: Dictionary with sequential skeletal data of the signing person
        :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
                            angle by which the landmarks will be rotated from
        :return: Dictionary with augmented (by rotation) sequential skeletal data of the signing person
        """

        #body_landmarks, hand_landmarks = self.__preprocess_row_sign(sign)
        angle = math.radians(random.uniform(*angle_range))

        body_landmarks = [[self.__rotate((0.5, 0.5), frame, angle) for frame in value] for value in
                        sign[:,self.BODY_IDENTIFIERS,:]]
        sign[:,self.BODY_IDENTIFIERS,:] = torch.tensor(body_landmarks)

        hand_landmarks = [[self.__rotate((0.5, 0.5), frame, angle) for frame in value] for value in
                        sign[:,self.HAND_IDENTIFIERS,:]]
        sign[:,self.HAND_IDENTIFIERS,:] = torch.tensor(hand_landmarks)

        return sign #self.__wrap_sign_into_row(body_landmarks, hand_landmarks)

    def augment_shear(self, sign: dict, type: str, squeeze_ratio: tuple) -> dict:
        """
        AUGMENTATION TECHNIQUE.
            - Squeeze. All the frames are squeezed from both horizontal sides. Two different random proportions up to 15% of
            the original frame's width for both left and right side are cut.
            - Perspective transformation. The joint coordinates are projected onto a new plane with a spatially defined
            center of projection, which simulates recording the sign video with a slight tilt. Each time, the right or left
            side, as well as the proportion by which both the width and height will be reduced, are chosen randomly. This
            proportion is selected from a uniform distribution on the [0; 1) interval. Subsequently, the new plane is
            delineated by reducing the width at the desired side and the respective vertical edge (height) at both of its
            adjacent corners.
        :param sign: Dictionary with sequential skeletal data of the signing person
        :param type: Type of shear augmentation to perform (either 'squeeze' or 'perspective')
        :param squeeze_ratio: Tuple containing the relative range from what the proportion of the original width will be
                            randomly chosen. These proportions will either be cut from both sides or used to construct the
                            new projection
        :return: Dictionary with augmented (by squeezing or perspective transformation) sequential skeletal data of the
                signing person
        """

        #body_landmarks, hand_landmarks = self.__preprocess_row_sign(sign)

        if type == "squeeze":
            move_left = random.uniform(*squeeze_ratio)
            move_right = random.uniform(*squeeze_ratio)

            src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)
            dest = np.array(((0 + move_left, 1), (1 - move_right, 1), (0 + move_left, 0), (1 - move_right, 0)),
                            dtype=np.float32)
            mtx = cv2.getPerspectiveTransform(src, dest)

        elif type == "perspective":

            move_ratio = random.uniform(*squeeze_ratio)
            src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)

            if self.__random_pass(0.5):
                dest = np.array(((0 + move_ratio, 1 - move_ratio), (1, 1), (0 + move_ratio, 0 + move_ratio), (1, 0)),
                                dtype=np.float32)
            else:
                dest = np.array(((0, 1), (1 - move_ratio, 1 - move_ratio), (0, 0), (1 - move_ratio, 0 + move_ratio)),
                                dtype=np.float32)

            mtx = cv2.getPerspectiveTransform(src, dest)

        else:

            logging.error("Unsupported shear type provided.")
            return {}


        landmarks_array = sign[:,self.BODY_IDENTIFIERS,:]#self.__dictionary_to_numpy(body_landmarks)
        augmented_landmarks = cv2.perspectiveTransform(np.array(landmarks_array, dtype=np.float32), mtx)

        augmented_zero_landmark = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), mtx)[0][0]
        augmented_landmarks = np.stack([np.where(sub == augmented_zero_landmark, [0, 0], sub) for sub in augmented_landmarks])
        sign[:,self.BODY_IDENTIFIERS,:] = torch.tensor(augmented_landmarks)
        #body_landmarks = self.__numpy_to_dictionary(augmented_landmarks)

        return sign#self.__wrap_sign_into_row(body_landmarks, hand_landmarks)


    def augment_arm_joint_rotate(self, sign: dict, probability: float, angle_range: tuple) -> dict:
        """
        AUGMENTATION TECHNIQUE. The joint coordinates of both arms are passed successively, and the impending landmark is
        slightly rotated with respect to the current one. The chance of each joint to be rotated is 3:10 and the angle of
        alternation is a uniform random angle up to +-4 degrees. This simulates slight, negligible variances in each
        execution of a sign, which do not change its semantic meaning.
        :param sign: Dictionary with sequential skeletal data of the signing person
        :param probability: Probability of each joint to be rotated (float from the range [0, 1])
        :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
                            angle by which the landmarks will be rotated from
        :return: Dictionary with augmented (by arm joint rotation) sequential skeletal data of the signing person
        """

        body_landmarks, hand_landmarks = self.__preprocess_row_sign(sign)

        # Iterate over both directions (both hands)
        for arm_side_ids in self.ARM_IDENTIFIERS_ORDER:
            for landmark_index, landmark_origin in enumerate(arm_side_ids):

                if self.__random_pass(probability):
                    angle = math.radians(random.uniform(*angle_range))

                    for to_be_rotated in arm_side_ids[landmark_index + 1:]:
                        augmented_values = [self.__rotate(sign[frame_index,landmark_origin,:], frame, angle) for frame_index, frame in enumerate(sign[:,to_be_rotated,:])]
                        augmented_values = torch.tensor(augmented_values)
                        sign[:,to_be_rotated,:] = augmented_values
        
        return sign


    if __name__ == "__main__":
        pass

