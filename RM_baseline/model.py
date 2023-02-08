import torch
from torch import nn

class DoraNet(nn.Module):
    def __init__(self,test=False):
        super(DoraNet, self).__init__()
        self.test=test
        self.mlp = nn.Sequential(
                nn.Linear(2,256),
                nn.ReLU(),
                nn.Linear(256,1024),
                nn.ReLU(),
                nn.Linear(1024,256),
                nn.ReLU(),
                nn.Linear(256, 4)
                )
                
    def forward(self, pos):
        pathloss = self.mlp(pos)

        return pathloss

        
def main():    
    b = 10
    doraNet = DoraNet()
    pos=torch.zeros((b,2))
    pathloss=torch.zeros(b,4)

    p_pathloss = doraNet(pos)
    print(torch.mean(torch.abs(p_pathloss-pathloss)))

        
if __name__ == '__main__':
    main()