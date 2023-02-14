import pandas as pd
import numpy as np
import torch

data_1 = np.load("data/train/user_01.npy")
data = np.reshape(data_1, (-1, 6))
print(data[0])
