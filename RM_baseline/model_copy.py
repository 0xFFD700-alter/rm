import torch
from torch import nn

class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 256),
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
            nn.Linear(2, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 32),
            nn.PReLU(),
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
        min = torch.tensor([-19.999460126918102, -20.000261487043904], dtype=pos.dtype, device=pos.device)
        max = torch.tensor([350.2780399339964, 352.50387612831156], dtype=pos.dtype, device=pos.device)
        pos = (2 * pos - max - min) / (max - min)
        reg = self.regnet(pos)
        cls = self.clsnet(pos)
        if self.training:
            return reg, cls
        else:
            return reg * (torch.sigmoid(cls) > 0.5).float() * (158.7472538974973 - 57.70791516029391) - 158.7472538974973

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