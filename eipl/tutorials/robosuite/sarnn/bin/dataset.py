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

def make_data(path,data_directory,data_volume,task,device = "cuda:1"):
    if "sorting" in task:
        #if task == "sorting_2_boxes":
        #data_directory.replace("_2_boxes","")
        state_files = np.load(sim_framework_path("/home/ito/d3il/environments/dataset/data/sorting/2_boxes_train_files.pkl"), allow_pickle=True)
    else:
        state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)
    joint_path = path + "all_data"
    bp_cam_imgs = []
    robot_joint_state_list = []
    
    print("len(path)",len(path))
    print("data_directory",data_directory)
    print("len(state_files)",len(state_files))
    #print("state files",state_files)

    for file in tqdm(state_files[:data_volume]):
        file_name = os.path.basename(file).split('.')[0]
        if file_name in ["env_1038_00","env_1179_00","env_1276_00","env_0268_00"]:
            #なぜか欠けてる?該当データ存在しない
            continue

        if task == "aligning":
            bp_imgs = glob.glob(joint_path+ '/images/bp-cam/' + file_name + '/*')
            with open(os.path.join(joint_path, 'state', file), 'rb') as f:
                env_state = pickle.load(f)

        elif task == "stacking":
            
            if "eval" in data_directory:
                #print(file_name)
                bp_imgs = glob.glob(path+ '/vision_data/images/bp-cam/' + file_name + '/*')
                with open(os.path.join(path, 'all_data', file), 'rb') as f:
                    env_state = pickle.load(f)
            else:
                bp_imgs = glob.glob(path+ '/vision_data/images/bp-cam/' + file_name + '/*')
                with open(os.path.join(path, 'vision_data/state', file), 'rb') as f:
                    env_state = pickle.load(f)

        elif "sorting" in task:
            path = "/home/ito/d3il/environments/dataset/data/sorting/"
            if "2" in task:
                bp_imgs = glob.glob(path+ '/2_boxes/images/bp-cam/' + file_name + '/*')
                with open(os.path.join(path, '2_boxes/state', file), 'rb') as f:
                    env_state = pickle.load(f)
            elif "4" in task:
                
                file_name = list(file_name)
                file_name.insert(-6,("0"))
                file_name = "".join(file_name)
                #print("bp_imgs",path+ '/4_boxes/images/bp-cam/' + file_name + '/*')
                bp_imgs = glob.glob(path+ '/4_boxes/images/bp-cam/' + file_name + '/*')
                
                
                with open(os.path.join(path, '4_boxes/state', file_name + ".pkl",), 'rb') as f:
                    env_state = pickle.load(f)
            elif "6" in task:
                #print("file_name",file_name)
                if "268" in file_name:
                    continue
                file_name = list(file_name)
                file_name.insert(-6,("0"))
                file_name = "".join(file_name)
                bp_imgs = glob.glob(path+ '/6_boxes/images/bp-cam/' + file_name + '/*')
                with open(os.path.join(path, '6_boxes/state', file_name+".pkl"), 'rb') as f:
                    env_state = pickle.load(f)
        
        #関節角度追加
        robot_joint_state = torch.from_numpy(env_state['robot']['j_pos'])
        robot_joint_state_list.append(robot_joint_state)


        bp_images = []
        
        #assert False
        bp_imgs.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

        for img in bp_imgs:
            image = cv2.imread(img).astype(np.float32)
            image = image.transpose((2, 0, 1)) / 255.

            image = torch.from_numpy(image).to(device).float().unsqueeze(0)

            bp_images.append(image)

        bp_images = torch.concatenate(bp_images, dim=0)
        bp_cam_imgs.append(bp_images)

    #print(bp_imgs)

    #長さが違うので0padding
    a = nn.utils.rnn.pack_sequence(bp_cam_imgs, enforce_sorted=False)
    padded_bp_cam_imgs, _ = nn.utils.rnn.pad_packed_sequence(a,batch_first =True,padding_value=0)
    
    tc = nn.utils.rnn.pack_sequence(robot_joint_state_list, enforce_sorted=False)
    padded_robot_joint_state, _ = nn.utils.rnn.pad_packed_sequence(tc,batch_first =True,padding_value=0)

    return padded_bp_cam_imgs,padded_robot_joint_state

"""
#リモートワーク用
#if args.aist:
path="/home/ito/d3il/environments/dataset/data/aligning/"
#else:
#    path="/home/ogata/workspace/ito/d3il/environments/dataset/data/aligning/"


import matplotlib.pyplot as plt

train_data_volume = 10
device = "cuda:0"


# 2x2のサブプロットを作成
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 10))
# 2×2のaxオブジェクトを一次元配列に変換
axes = axes.flatten()

train_img,train_joint = make_data(path,path+"train_files.pkl",train_data_volume,device="cpu")


print(len(train_joint))
print("*****")

# それぞれのサブプロットにランダムデータを使用して画像を描画
for i in range(10):
        # axesから各axオブジェクトを取得
        ax = axes[i]
        # ランダムなデータを生成
        #print(len(train_joint[:][i]))

        #     print(train_joint[2].shape)
        #data = train_joint[:][i]
        #print(len(train_joint[0][0][i]))
        print(len(train_joint[:][i]),len(range(242)))
        ax.scatter(train_joint[:][i],range(243))
        #train_joint[0][:][])
        # サブプロットに画像を描画
        #img = ax.imshow(data)
        # カラーバーの追加
        #cbar = fig.colorbar(img, ax=ax)
        #cbar.set_label('Value')
        # タイトルを追加
        #ax.set_title(f'Image at {i}')
        # 軸の目盛りを非表示
        #ax.axis('off')

# サブプロット間のスペースを調整
plt.tight_layout()
plt.savefig('./bin/dataplot.png')
#plt.show()


print("path",path)
print(train_joint.shape,len(train_joint))
#plt.scatter(train_joint[0][:][], train_joint[0][][])

"""