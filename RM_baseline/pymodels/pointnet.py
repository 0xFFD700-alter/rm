import torch
from torch import nn
from torch.nn import functional as F
# import numpy as np

INPUT_DIM = 8
OUTPUT_DIM = 4
N_PTS = 32

class DoraNetfeat(nn.Module):
    def __init__(self):
        super(DoraNetfeat, self).__init__()
        self.conv1 = nn.Conv1d(INPUT_DIM, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.max_pool1d(x, x.size(dim=-1))
        x = x.repeat(1, 1, pointfeat.size(dim=-1))
        return torch.cat([x, pointfeat], 1)

class binary_classifier_mlp(nn.Module):
    def __init__(self, dim):
        super(binary_classifier_mlp, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 32),
            nn.PReLU(),
            nn.Linear(32, 4)
        )
                
    def forward(self, x):
        x = self.mlp(x)
        return x

class DoraNet(nn.Module):
    def __init__(self, k=OUTPUT_DIM):
        super(DoraNet, self).__init__()
        self.k = k
        self.feat = DoraNetfeat()
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.classifier = binary_classifier_mlp(INPUT_DIM)

    def _preprocess(self, x):
        p = torch.tensor([
            [65.78919014, 65.21920509],
            [65.06103647, 264.39978533],
            [265.22510056, 65.55262073],
            [264.8003945, 265.14501625]
        ], dtype=x.dtype, device=x.device)
        min = torch.tensor([-11.25605105, -52.85139191], dtype=x.dtype, device=x.device)
        max = torch.tensor([359.02144901, 319.6527457], dtype=x.dtype, device=x.device)
        x = (2 * x - max - min) / (max - min)
        p = (2 * p - max - min) / (max - min)
        x = torch.cat([x - p[i, :] for i in range(4)], 1)
        return x
    
    def _postprocess(self, x):
        pass

    def _interpolate(self, x):
        pass

    def _forward_features(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

    def forward(self, x):
        x = self._preprocess(x)
        mask = self.classifier(x)
        if self.training:
            x = x.reshape(-1, N_PTS, INPUT_DIM).transpose(1, 2)
            x = self._forward_features(x)
            return x.transpose(1, 2).reshape(-1, OUTPUT_DIM), mask
        else:
            x = x.unsqueeze(-1)
            x = self._forward_features(x)
            return x.squeeze() * (torch.sigmoid(mask) > 0.5).float()

def main():    
    b = 64
    model = DoraNet()
    x = torch.zeros((b, 2))
    pathloss = torch.zeros((b, 4))
    cls = torch.zeros((b, 4))

    model.train()
    p_pathloss, p_cls = model(x)
    print(torch.mean(torch.abs(p_pathloss - pathloss)), torch.mean(torch.abs(p_cls - cls)))

    x = torch.zeros((1, 2))
    model.eval()
    p_pathloss = model(x)
    print(torch.mean(torch.abs(p_pathloss - pathloss)))
        
if __name__ == '__main__':
    main()


# class STNkd(nn.Module):
#     def __init__(self, k=INPUT_DIM):
#         super(STNkd, self).__init__()
#         self.conv1 = nn.Conv1d(k, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k*k)
#         self.relu = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)

#         self.k = k

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x

# class PointNetfeat(nn.Module):
#     def __init__(self, global_feat = True, feature_transform = False):
#         super(PointNetfeat, self).__init__()
#         self.stn = STNkd(k=INPUT_DIM)
#         self.conv1 = nn.Conv1d(INPUT_DIM, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=64)

#     def forward(self, x):
#         n_pts = x.size()[2]
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         x = torch.bmm(x, trans)
#         x = x.transpose(2, 1)
#         x = F.relu(self.bn1(self.conv1(x)))

#         if self.feature_transform:
#             trans_feat = self.fstn(x)
#             x = x.transpose(2, 1)
#             x = torch.bmm(x, trans_feat)
#             x = x.transpose(2, 1)
#         else:
#             trans_feat = None

#         pointfeat = x
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         if self.global_feat:
#             return x, trans, trans_feat
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
#             return torch.cat([x, pointfeat], 1), trans, trans_feat

# class PointNetDenseCls(nn.Module):
#     def __init__(self, k=OUTPUT_DIM, feature_transform=False):
#         super(PointNetDenseCls, self).__init__()
#         self.k = k
#         self.feature_transform=feature_transform
#         self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
#         self.conv1 = nn.Conv1d(1088, 512, 1)
#         self.conv2 = nn.Conv1d(512, 256, 1)
#         self.conv3 = nn.Conv1d(256, 128, 1)
#         self.conv4 = nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)

#     def forward(self, x):
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
#         x = x.transpose(2,1).contiguous()
#         x = F.log_softmax(x.view(-1,self.k), dim=-1)
#         x = x.view(batchsize, n_pts, self.k)
#         return x, trans, trans_feat

# def feature_transform_regularizer(trans):
#     d = trans.size()[1]
#     batchsize = trans.size()[0]
#     I = torch.eye(d)[None, :, :]
#     if trans.is_cuda:
#         I = I.cuda()
#     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
#     return loss