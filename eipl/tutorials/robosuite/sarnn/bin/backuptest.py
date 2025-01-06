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

"""
# load dataset
minmax = [params["vmin"], params["vmax"]]
images_raw = np.load("../simulator/data/test/images.npy")
joints_raw = np.load("../simulator/data/test/joints.npy")
joint_bounds = np.load("../simulator/data/joint_bounds.npy")
images = images_raw[idx]
joints = joints_raw[idx]
"""
#print(images.shape)
#print(images)
#print(joints.shape)
#print(joints)

#assert False

#リモートワーク用
path="/home/ito/d3il/environments/dataset/data/aligning/"
#path="/home/ogata/workspace/ito/d3il/environments/dataset/data/aligning/"
device = "cuda:0"
#images_1,joints_1 = make_data(path,path+"eval_files.pkl",device=device)
images,joints = make_data(path,path+"eval_files.pkl",30,device=device)

#images = images_1[0]
#joints = joints_1[0]
print("images.shape",images.shape)
print("joints.shape",joints.shape)
print("joints",joints)
#assert False

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
model.eval()
model.to(device)


print("--------")
print("images",len(images))
# Inference
im_size = 96
image_list, joint_list = [], []
ect_pts_list, dec_pts_list = [], []
state = None
nloop = len(images)
start = 0
nloop = 25
print("imags",images.shape)

for loop_ct in range(start,start+nloop):
    # load data and normalization
    #img_t = images[loop_ct].transpose(2, 0, 1)
    print("loopct",loop_ct)
    """
    img_t = normalization(img_t, (0, 255), minmax)
    img_t = torch.Tensor(np.expand_dims(img_t, 0))
    joint_t = normalization(joints[loop_ct], joint_bounds, minmax)
    joint_t = torch.Tensor(np.expand_dims(joint_t, 0))
    """

    img_t = images[loop_ct].to(torch.float32).to(device)
    joint_t = joints[loop_ct].to(torch.float32).to(device)
    # predict rnn
    y_image, y_joint, ect_pts, dec_pts, state = model(img_t, joint_t, state)

    img_t.to("cpu")
    joint_t.to("cpu")
    # denormalization
    #pred_image = tensor2numpy(y_image[0].permute(1,2,0))
    pred_image = tensor2numpy(y_image[loop_ct].permute(1,2,0))
    
    
    #pred_image = deprocess_img(pred_image, params["vmin"], params["vmax"])
    #pred_image = pred_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[loop_ct])
    #pred_joint = normalization(pred_joint, minmax, joint_bounds)

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)
    ect_pts_list.append(tensor2numpy(ect_pts[0]))
    dec_pts_list.append(tensor2numpy(dec_pts[0]))

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

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
#images = images.permute(0,3,2,1)
T = len(images)
fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=60)

print("---------------")
print("image_list",len(image_list))
print("image_list[0]",len(image_list[0]))
print("pred images",len(pred_image))
print("pred images[0]",len(pred_image[0]))
print("images",len(images))
print("images[0]",len(images[0]))
images = images[0].permute(0,2,3,1)

def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot camera image
    #print("images",images[i].shape)
    ax[0].imshow(images[i].to('cpu').detach().numpy().copy())
    #ax[0].imshow(images[i].permute(1,2,0).to('cpu').detach().numpy().copy())
    #ax[0].imshow(images[0][i].to('cpu').detach().numpy().copy())

    for j in range(params["k_dim"]):
        ax[0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1], "co", markersize=12)  # encoder
        ax[0].plot(
            dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=12, markeredgewidth=2
        )  # decoder
    ax[0].axis("off")
    ax[0].set_title("Input image", fontsize=20)

    # plot predicted image
    ax[1].imshow(pred_image[i])
    ax[1].axis("off")
    ax[1].set_title("Predicted image", fontsize=20)

    # plot joint angle
    ax[2].set_ylim(-np.pi, 4)
    ax[2].set_xlim(0, T)
    ax[2].plot(joints[1:][0], linestyle="dashed", c="k")
    # om has 5 joints, not 8
    for joint_idx in range(7):
        ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[2].set_xlabel("Step", fontsize=20)
    ax[2].set_title("Joint angles", fontsize=20)
    ax[2].tick_params(axis="x", labelsize=16)
    ax[2].tick_params(axis="y", labelsize=16)
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.12, top=0.9)


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
#ani.save("./output/SARNN_{}_{}.gif".format(params["tag"], idx))

ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx,"1"), writer="ffmpeg")
