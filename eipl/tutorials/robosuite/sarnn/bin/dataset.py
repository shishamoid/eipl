from sim_path import sim_framework_path
import torch
import os
import glob
import tqdm
import pickle
from tqdm import tqdm

import torch.nn as nn
import numpy as np
import cv2

def make_data(path,data_directory,device = "cuda:1"):
    state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)
    path = path + "all_data"
    bp_cam_imgs = []
    robot_joint_state_list = [] 

    for file in tqdm(state_files[:3]):
        with open(os.path.join(path, 'state', file), 'rb') as f:
            env_state = pickle.load(f)

        #関節角度追加
        robot_joint_state = torch.from_numpy(env_state['robot']['j_pos'])
        robot_joint_state_list.append(robot_joint_state)

        file_name = os.path.basename(file).split('.')[0]

        bp_images = []
        bp_imgs = glob.glob(path+ '/images/bp-cam/' + file_name + '/*')
        bp_imgs.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

        for img in bp_imgs:
            image = cv2.imread(img).astype(np.float32)
            image = image.transpose((2, 0, 1)) / 255.

            image = torch.from_numpy(image).to(device).float().unsqueeze(0)

            bp_images.append(image)

        bp_images = torch.concatenate(bp_images, dim=0)
        bp_cam_imgs.append(bp_images)

    #長さが違うので0padding
    a = nn.utils.rnn.pack_sequence(bp_cam_imgs, enforce_sorted=False)
    padded_bp_cam_imgs, _ = nn.utils.rnn.pad_packed_sequence(a,batch_first =True,padding_value=0)
    
    tc = nn.utils.rnn.pack_sequence(robot_joint_state_list, enforce_sorted=False)
    padded_robot_joint_state, _ = nn.utils.rnn.pad_packed_sequence(tc,batch_first =True,padding_value=0)

    return padded_bp_cam_imgs,padded_robot_joint_state
