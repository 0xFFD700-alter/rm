import torch
import torch.nn as nn
import math

class AttentionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.attention = nn.Linear(output_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        attention_weights = self.attention(x)
        x = torch.mul(x, attention_weights)
        return x


class TransformerNet(nn.Module):
    def __init__(self, input_dim):
        super(TransformerNet, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_dim, 1), 2)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(2, 2), 2)
        self.linear = nn.Linear(2, 4)

    def forward(self, x):
        # x.view(1, -1, 2)
        x = self.encoder(x)
        memory = torch.randn(50, 2).to(device="cuda:0")
        x = self.decoder(x, memory)
        x = self.linear(x)
        return x


# 设置参数
input_dim = 2
hidden_dim = 8
output_dim = 4
num_layers = 2

# Transformer网络
model = nn.Sequential()

# 第一层：输入层
model.add_module('input_layer', nn.Linear(input_dim, hidden_dim))

# 第二层至最后一层：多层Transformer
for i in range(num_layers):
    model.add_module(f'transformer_{i}', nn.Transformer(hidden_dim, hidden_dim))

# 最后一层：输出层
model.add_module('output_layer', nn.Linear(hidden_dim, output_dim))

# 输入数据
X = torch.randn(2, 2)

# 运行模型得到输出
output = model(X)

print(output)
