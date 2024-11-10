#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization
from eipl.model import SARNN

from dataset import make_data

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
parser.add_argument("--idx", type=int, default=0)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = args.idx

minmax = [params["vmin"], params["vmax"]]

#リモートワーク用
path="/home/ito/d3il/environments/dataset/data/aligning/"
#path="/home/ogata/workspace/ito/d3il/environments/dataset/data/aligning/"
device = "cuda:0"
device = "cpu"
#images,joints = make_data(path,path+"eval_files.pkl",100,device=device)
images_1,joints_1 = make_data(path,path+"eval_files.pkl",40,device=device)

images = images_1[9]
joints = joints_1[9]

# define model
model = SARNN(
    rec_dim=params["rec_dim"],
    joint_dim=7,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
    im_size=[96, 96],
)

if params["compile"]:
    model = torch.compile(model)

# load weight
#ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
ckpt = torch.load(args.filename, map_location=torch.device("cuda:0"))

model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

# Inference
im_size = 96
image_list, joint_list = [], []
ect_pts_list, dec_pts_list = [], []
state = None
#nloop = len(images)
start = 0
last = 150
print("----------------")
print("images",images.shape)


for loop_ct in range(start,last):
    img_t = images[loop_ct].to(torch.float32).to(device).unsqueeze(dim=0)
    joint_t = joints[loop_ct].to(torch.float32).to(device).unsqueeze(dim=0)
    # predict rnn
    y_image, y_joint, ect_pts, dec_pts, state = model(img_t, joint_t, state)
    #print("y_image.shape",y_image.shape) #[1,3,96,96]
    #print

    pred_image = tensor2numpy(y_image[0].permute(1,2,0))
    pred_joint = tensor2numpy(y_joint[0])
    
    #pred_joint = normalization(pred_joint, minmax, joint_bounds)

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)
    ect_pts_list.append(tensor2numpy(ect_pts[0]))
    dec_pts_list.append(tensor2numpy(dec_pts[0]))

    #print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_image = np.array(image_list)
pred_joint = np.array(joint_list)

# split key points
ect_pts = np.array(ect_pts_list)
dec_pts = np.array(dec_pts_list)
ect_pts = ect_pts.reshape(-1, params["k_dim"], 2) * im_size
dec_pts = dec_pts.reshape(-1, params["k_dim"], 2) * im_size
enc_pts = np.clip(ect_pts, 0, im_size)
dec_pts = np.clip(dec_pts, 0, im_size)


# plot images
#T = len(images)
T = last-start

fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=60)

def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(images[i].permute(1,2,0).to('cpu').detach().numpy().copy()*0.5+0.5)

    for j in range(params["k_dim"]):
        ax[0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1], "co", markersize=12)  # encoder
        ax[0].plot(
            dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=12, markeredgewidth=2
        )  # decoder
    ax[0].axis("off")
    ax[0].set_title("Input image", fontsize=20)

    # plot predicted image
    ax[1].imshow(pred_image[i]*0.3+0.5)
    ax[1].axis("off")
    ax[1].set_title("Predicted image", fontsize=20)

    # plot joint angle
    ax[2].set_ylim(-np.pi, 3.4)
    ax[2].set_xlim(0, T)
    ax[2].plot(joints[1:], linestyle="dashed", c="k")
    # om has 5 joints, not 8
    
    for joint_idx in range(7):
        ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[2].set_xlabel("Step", fontsize=20)
    ax[2].set_title("Joint angles", fontsize=20)
    ax[2].tick_params(axis="x", labelsize=16)
    ax[2].tick_params(axis="y", labelsize=16)
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.12, top=0.9)


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 1000)), frames=T)
#print(ani.shape)
ani.save("./output/SARNN_{}_{}.gif".format(params["tag"], idx))

# If an error occurs in generating the gif animation, change the writer (imagemagick/ffmpeg).
# ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx, args.input_param), writer="ffmpeg")
