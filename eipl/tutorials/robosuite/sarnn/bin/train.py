#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from eipl.model import SARNN
from eipl.data import MultimodalDataset
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization

# load own library
sys.path.append("./libs/")
from fullBPTT import fullBPTTtrainer

#0819追加
import glob
import cv2
import pickle
import torch.nn as nn

# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model", type=str, default="sarnn")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--rec_dim", type=int, default=50)
parser.add_argument("--k_dim", type=int, default=5)
parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=0.1)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)
parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)

# calculate the noise level (variance) from the normalized range
stdev = args.stdev * (args.vmax - args.vmin)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# load dataset
minmax = [args.vmin, args.vmax]
"""
images_raw = np.load("../simulator/data/train/images.npy")
joints_raw = np.load("../simulator/data/train/joints.npy")
joint_bounds = np.load("../simulator/data/joint_bounds.npy")
images = normalization(images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
joints = normalization(joints_raw, joint_bounds, minmax)
"""

#0819追加
#画像
data_dir = "/home/ogata/workspace/ito/d3il/environments/dataset/data/aligning/all_data/"
bp_images = []
bp_imgs = glob.glob(data_dir + 'images/bp-cam/*/*.jpg')
data_length = 10
#bp_imgs.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

counter = 0
for img in bp_imgs:
    counter +=1
    if counter ==data_length:
        print("img",img)
        break
    image = cv2.imread(img).astype(np.float32)
    image = image.transpose((2, 0, 1)) / 255.

    #image = torch.from_numpy(image).to(device).float().unsqueeze(0)

    bp_images.append(image)


#関節角度
state_list = glob.glob(data_dir+"state/*.pkl")
print(len(state_list))
robot_state_list = []

for i in range(len(state_list)):
    path = state_list[i]
    if i == data_length:
        break
    with open(path, 'rb') as f:
        env_state = pickle.load(f)
        robot_joint_state = torch.from_numpy(env_state['robot']['j_pos'])
    robot_state_list.append(robot_joint_state)

print(len(robot_state_list))

print("Dataset Loaded!!")
print("bp_images",len(bp_images))
print("robot_state_list",len(robot_state_list))
#assert False


#長さが違うので0padding
#self.data_length = 
a = nn.utils.rnn.pack_sequence(robot_state_list, enforce_sorted=False)
robot_state_list, _ = nn.utils.rnn.pad_packed_sequence(a,batch_first =True,padding_value=0)

train_dataset = MultimodalDataset(bp_images, robot_state_list, device=device, stdev=stdev)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
)

images_raw = np.load("../simulator/data/test/images.npy")
joints_raw = np.load("../simulator/data/test/joints.npy")
images = normalization(images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
joints = normalization(joints_raw, joint_bounds, minmax)
test_dataset = MultimodalDataset(images, joints, device=device, stdev=None)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
)

# define model
model = SARNN(
    rec_dim=args.rec_dim,
    joint_dim=8,
    k_dim=args.k_dim,
    heatmap_size=args.heatmap_size,
    temperature=args.temperature,
    im_size=[64, 64],
)

# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

# set optimizer
optimizer = optim.Adam(model.parameters(), eps=1e-07)

# load trainer/tester class
loss_weights = [args.img_loss, args.joint_loss, args.pt_loss]
trainer = fullBPTTtrainer(model, optimizer, loss_weights=loss_weights, device=device)

### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "SARNN.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch(train_loader)
        with torch.no_grad():
            test_loss = trainer.process_epoch(test_loader, training=False)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)

        # early stop
        save_ckpt, _ = early_stop(test_loss)

        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name)

        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss, test_loss=test_loss))
        pbar_epoch.update()
