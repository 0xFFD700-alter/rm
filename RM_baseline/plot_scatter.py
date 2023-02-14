"""
@Author : pfzhang
@Email  : pfzhang2022@shanghaitech.edu.cn
@Date   : 2023-02-12 16:31
@Desc   : 
"""

from dataset import DoraSet, DoraSetComb
import matplotlib.pyplot as plt
from model import DoraNet
import numpy as np
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
sys.path.append("../..")
if not os.path.exists(f'models/'):
    os.makedirs(f'models/')
if not os.path.exists(f'results/'):
    os.makedirs(f'results/')

train_dataset_path = "data/train/"
data_users = []
for i in range(1, 21):
    data_user = np.reshape(np.load('./data/train/user_01.npy'),(-1,6))
    data_users.append(data_user)

data_users = np.vstack(data_users)
coordinate_users = data_users[:,0:2]
