"""
@Author : pfzhang
@Email  : pfzhang2022@shanghaitech.edu.cn
@Date   : 2023-02-15 23:32
@Desc   :
"""
import copy
import sys
from model import DoraNet
from util import *
from dataset import DoraSet, DoraSetComb
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
sys.path.append("../..")

epochs = 500  # total epochs
local_epochs = 10 # local epochs of each user at an iteration
saveLossInterval = 1  # intervals to save loss
saveModelInterval = 10  # intervals to save model
batchSize = 512  # batchsize for training and evaluation
num_users = 90   # total users
num_activate_users = 5
lr = 3e-4  # learning rate
cudaIdx = "cuda:0"  # GPU card index
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
num_workers = 0  # workers for dataloader
evaluation = False  # evaluation only if True
criterion = torch.nn.MSELoss().to(device)


class Link(object):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.size = np.zeros((1,), dtype=np.float64)

    def pass_link(self, pay_load1, pay_load2):
        for k, v in pay_load1.items():
            self.size = self.size + np.sum(v.numel())
        for k, v in pay_load2.items():
            self.size = self.size + np.sum(v.numel())
        return pay_load1, pay_load2


class ScaffoldServer: # used as a center
    def __init__(self, global_parameters, global_control_parameters, down_link):
        self.global_parameters = global_parameters
        self.global_control_parameters = global_control_parameters
        self.down_link = down_link

    def download(self, user_idx):
        local_parameters = []
        local_control_parameters = []
        for i in range(len(user_idx)):
            local_parameter, local_control_parameter = self.down_link.pass_link(copy.deepcopy(self.global_parameters), copy.deepcopy(self.global_control_parameters))
            local_parameters.append(local_parameter)
            local_control_parameters.append(local_control_parameter)
        return local_parameters, local_control_parameters

    def upload(self, delta_parameters, delta_controls):
        for i, (k, v) in enumerate(self.global_parameters.items()):
            tmp_v = torch.zeros_like(v, dtype=torch.float)
            for j in range(len(delta_parameters)):
                tmp_v += delta_parameters[j][k]
            tmp_v = tmp_v / len(delta_parameters)
            self.global_parameters[k] = self.global_parameters[k] + tmp_v
        tmp_controls = {}
        for i, (k, v) in enumerate(self.global_parameters.items()):
            tmp_v = torch.zeros_like(v)
            for j in range(len(delta_controls)):
                tmp_v += delta_controls[j][k]
            tmp_v = tmp_v / len(delta_parameters)
            tmp_controls[k] = tmp_v
        return tmp_controls



class Client: # as a user
    def __init__(self, data_loader, user_idx):
        self.data_loader = data_loader
        self.user_idx = user_idx

    def train(self, model, learningRate, idx, global_model, global_control, local_control): # training locally
        optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)
        count = 0
        for local_epoch in range(1, local_epochs + 1):
            for i, (pos, pathloss) in enumerate(self.data_loader):
                pos = pos.float().to(device)
                pathloss = pathloss.float().to(device)
                mask = torch.where(pathloss != 0., 1., 0.).float().to(device)
                optimizer.zero_grad()
                reg, cls = model(pos)
                reg_loss = torch.mean(torch.abs(reg[pathloss != 0] - pathloss[pathloss != 0]))
                reg_loss.backward()
                cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls, mask)
                cls_loss.backward()
                optimizer.step()
                global_parameter = global_model.state_dict()
                local_parameter = model.state_dict()
                global_control_parameter = global_control.state_dict()
                local_control_parameter = local_control.state_dict()
                for name in local_parameter:
                    # 10 line in algorithm
                    local_parameter[name] = local_parameter[name] - learningRate *  (global_control_parameter[name] - local_control_parameter[name])
                # print(f"Client: {idx}({self.user_idx:2d}) Local Epoch: [{local_epoch}][{i+1}/{len(self.data_loader)}]---- loss {reg_loss.item():f}, {cls_loss.item():f}")
                model.load_state_dict(local_parameter)
                count = count + 1
        # local_control_parameters = local_control.state_dict()
        delta_parameter = copy.deepcopy(local_parameter)
        # local_parameters = model.state_dict()
        delta_control = copy.deepcopy(local_parameter)
        local_control_parameter_plus = copy.deepcopy(local_parameter)
        for name in local_control_parameter_plus:
            # line 12 in algo
            local_control_parameter_plus[name] = local_control_parameter[name] - global_control_parameter[name] + (global_parameter[name] - local_parameter[name]) / (count * learningRate)
            # line 13 in algo
            delta_control[name] = local_control_parameter_plus[name] - local_control_parameter[name]
            delta_parameter[name] = local_parameter[name] - global_parameter[name]
        return delta_parameter, delta_control, local_control_parameter_plus

def activateClient(train_dataloaders, user_idx, server):
    local_parameters, local_control_parameters = server.download(user_idx)
    clients = []
    for i in range(len(user_idx)):
        clients.append(Client(train_dataloaders[user_idx[i]], user_idx[i]))
    return clients, local_parameters


def train(train_dataloaders, user_idx, server, global_model,  global_control, local_controls, up_link, learningRate):
    clients, local_parameters = activateClient(train_dataloaders, user_idx, server)
    delta_parameters = [copy.deepcopy(local_parameters[i]) for i in range(len(user_idx))]
    delta_controls = [copy.deepcopy(local_parameters[i]) for i in range(len(user_idx))]
    for i in range(len(user_idx)):
        model = DoraNet().to(device)
        model.load_state_dict(local_parameters[i])
        model.train()
        delta_parameter, delta_control, local_control_parameters_plus = clients[i].train(model, learningRate, i, global_model, global_control, local_controls[user_idx[i]])
        local_controls[user_idx[i]].load_state_dict(local_control_parameters_plus)
        delta_parameters[i], delta_controls[i] = up_link.pass_link(delta_parameter, delta_control)
    tmp_control = server.upload(delta_parameters, delta_controls)
    global_control_parameter = global_control.state_dict()
    for name in global_control_parameter:
        global_control_parameter[name] = global_control_parameter[name] + tmp_control[name] * len(delta_parameters) / num_users
    global_control.load_state_dict(global_control_parameter)
    global_model.load_state_dict(server.global_parameters)


def valid(data_loader, model, epoch):
    with torch.no_grad():
        model.eval()
        losses = Recoder()
        scores = Recoder()
        for i, (pos, pathloss) in enumerate(data_loader):
            pos = pos.float().to(device)
            pathloss = pathloss.float().to(device)
            p_pathloss = model(pos)
            loss = torch.mean(torch.abs(p_pathloss[pathloss != 0] - pathloss[pathloss != 0])) ## unit in dB
            tmp1 = torch.sum(torch.abs(10 ** (0.1 * p_pathloss[pathloss != 0]) - 10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
            tmp2 = torch.sum(torch.abs(10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
            score = tmp1 / tmp2
            if score>1:
                score=torch.tensor([1])
            losses.update(loss.item(), len(pos))
            scores.update(score.item(), len(pos))
        print(f"Global Epoch: {epoch}----loss:{losses.avg():.4f}----pathloss_score:{-10 * np.log10(scores.avg()):.4f}")
    return -10 * np.log10(scores.avg())


def train_main(train_dataset_path):
    seed_everything(42)
    train_dataloaders = []
    train_datasets = []
    valid_datasets = []
    if not os.path.exists(f'models/'):
        os.makedirs(f'models/')
    if not os.path.exists(f'results/'):
        os.makedirs(f'results/')
    for i in range(1, num_users + 1):
        all_dataset = DoraSet(train_dataset_path, set='train', clientId=i)
        train_size = int(0.99 * len(all_dataset))
        valid_size = len(all_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, [train_size, valid_size])
        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batchSize, shuffle=True, num_workers=num_workers, drop_last=True)
        train_dataloaders.append(train_loader)

    valid_data_comb = DoraSetComb(valid_datasets)
    valid_loader = torch.utils.data.DataLoader(valid_data_comb, 1, shuffle=False, num_workers=num_workers)
    global_model = DoraNet().to(device)
    global_control = DoraNet().to(device)
    global_model_parameter = global_model.state_dict()
    global_control_parameter = global_control.state_dict()
    local_controls = [DoraNet().to(device) for i in range(num_users)]
    for net in local_controls:
        net.load_state_dict(global_control_parameter)
    up_link = Link("uplink")
    down_link = Link("downlink")
    server = ScaffoldServer(global_model_parameter, global_control_parameter, down_link)

    pathloss_scores = []
    ul_commCost_scores = []
    dl_commCost_scores = []
    for epoch in range(1, epochs + 1):  ## start training
        user_idx = np.random.choice(a=num_users, size=num_activate_users, replace=False, p=None).tolist()
        train(train_dataloaders, user_idx, server, global_model, global_control, local_controls, up_link, lr)
        test_model = copy.deepcopy(global_model).to(device)
        pathloss_score = valid(valid_loader, test_model, epoch)
        pathloss_scores.append(pathloss_score)
        ul_commCost_scores.append(up_link.size)
        dl_commCost_scores.append(down_link.size)
        checkPoint(epoch, epochs, global_model, pathloss_scores, ul_commCost_scores, dl_commCost_scores, saveModelInterval, saveLossInterval)


if __name__ == '__main__':
    train_main("data/train/")