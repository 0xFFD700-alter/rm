import torch
from torch import nn

class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
                
    def forward(self, pos):
        return self.mlp(pos)

class ClsNet(nn.Module):
    def __init__(self):
        super(ClsNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
                
    def forward(self, pos):
        return self.mlp(pos)

class DoraNet(nn.Module):
    def __init__(self):
        super(DoraNet, self).__init__()
        self.regnet = RegNet()
        self.clsnet = ClsNet()

    def forward(self, pos):
        bsp = torch.tensor([
            [65.78919014, 65.21920509],
            [65.06103647, 264.39978533],
            [265.22510056, 65.55262073],
            [264.8003945, 265.14501625]
        ], dtype=pos.dtype, device=pos.device)
        max = torch.tensor([479.71103286 for _ in range(4)], dtype=pos.dtype, device=pos.device)
        pos = torch.cat([torch.norm(pos - bsp[i, :], p=2, dim=1, keepdim=True) for i in range(4)], 1)
        pos = pos / max
        reg = self.regnet(pos)
        cls = self.clsnet(pos)
        if self.training:
            return reg, cls
        else:
            return (reg * (158.7472538974973 - 57.70791516029391) - 158.7472538974973) * (torch.sigmoid(cls) > 0.5).float()

def main():    
    b = 1
    model = DoraNet()
    pos = torch.zeros((b, 2))
    reg = torch.zeros((b, 4))
    cls = torch.zeros((b, 4))

    model.train()
    p_reg, p_cls = model(pos)
    print(torch.mean(torch.abs(p_reg - reg)), torch.mean(torch.abs(p_cls - cls)))

    model.eval()
    p_reg = model(pos)
    print(torch.mean(torch.abs(p_reg - reg)))
        
if __name__ == '__main__':
    main()