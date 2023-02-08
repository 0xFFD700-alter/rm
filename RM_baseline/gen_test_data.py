import json
import os
import numpy as np

num_env=1
test_instances=10

if not os.path.exists("data_test/label"):
    os.makedirs("data_test/label")
if not os.path.exists("data_test/test_input"):
    os.makedirs("data_test/test_input")
if not os.path.exists("data_test/test_output"):
    os.makedirs("data_test/test_output")

path="data/train/"
for i in range(test_instances):
    envPath = path
    data = np.reshape(np.load(envPath + 'user_01.npy'),(-1,6))
    pos=data[i,:2]
    pathloss=data[i,2:]
    filename = 'data_test/label/label_' + str(i) + '.txt'
    with open(filename, 'w') as name:
        data = {'pathloss': pathloss.tolist()}
        jsonData = json.dumps(data, ensure_ascii=False)
        name.write(jsonData)

    filename = 'data_test/test_input/data_'+str(i)+'.txt'
    with open(filename, 'w') as name:
        data = {'pos': pos.tolist()}
        jsonData = json.dumps(data, ensure_ascii=False)
        name.write(jsonData)



