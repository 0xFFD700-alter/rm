import torch
import argparse
import torch.nn as nn
from neuron import IFNeuron, rate_spikes
from spikingjelly.clock_driven import neuron, encoding, functional
parser = argparse.ArgumentParser(description='6G SNN')
parser.add_argument('-T', '--timesteps', default=30, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--train_Vth', default=1, type=int)
parser.add_argument('-V', '--vth', default=1.0, type=float, help='膜电压阈值')
parser.add_argument('--Vth_bound', default=0.0005, type=float)
parser.add_argument('--alpha', default=0.3, type=float)
parser.add_argument('--rate_stat', default=0, type=int)

args = parser.parse_args()
snn_setting = {}
snn_setting['timesteps'] = args.T
snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
snn_setting['Vth'] = args.vth
snn_setting['alpha'] = args.alpha
snn_setting['Vth_bound'] = args.Vth_bound
snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
encoder = encoding.PoissonEncoder()
class DoraNet(nn.Module):
    def __init__(self, snn_setting, test=False):
        super().__init__()
        self.test = test
        self.setting = snn_setting
        self.FC1 = nn.Linear(2, 256, bias=False)
        self.if1 = IFNeuron(self.setting)
        self.FC2 = nn.Linear(256, 1024, bias=False)
        self.if2 = IFNeuron(self.setting)
        self.FC3 = nn.Linear(1024, 256, bias=False)
        self.if3 = IFNeuron(self.setting)
        self.FC4 = nn.Linear(256, 4, bias=False)
        # self.if4 = IFNeuron(self.setting)


    def forward(self, x):
        input_img = []
        for t in range(args.T):
            input_img.append(encoder(x).float())
        input_tensor = torch.cat(input_img, dim=0)
        out = self.FC1(input_tensor)
        out = self.if1(out)
        out = self.FC2(out)
        out = self.if2(out)
        out = self.FC3(out)
        out = self.if3(out)
        out = self.FC4(out)
        out = torch.sum(out.view(args.T, -1, 4), 0) / args.T
        return out


def main():
    b = 10
    doraNet = DoraNet(snn_setting)
    pos = torch.zeros((b, 2))
    pathloss = torch.zeros(b, 4)

    p_pathloss = doraNet(pos)
    print(torch.mean(torch.abs(p_pathloss - pathloss)))


if __name__ == '__main__':
    main()