import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Any


def rate_spikes(data, timesteps):  # 返回时间步长timestep内的发放率，针对IF神经元
    chw = data.size()[1:]  # 获取data的channel，height，width，返回一个tensor
    firing_rate = torch.mean(data.view(timesteps, -1, *chw), 0)  # 数据维度变成（timestep，batch，channel，height，width），按照第一维度，也就是步长求平均
    return firing_rate


def sum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


class IFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, timesteps=10, Vth=1.0, alpha=0.5):
        ctx.save_for_backward(input)

        chw = input.size()[1:]
        size = input.size()
        input_reshape = input.view(timesteps, -1, *chw)   # 输入电流形状重塑成（timestep，batch，channel，height，width）
        mem_potential = torch.zeros(input_reshape.size(1), *chw).to(input_reshape.device)  # 初始膜电压设置为全0
        spikes = []

        for t in range(input_reshape.size(0)):  # 根据时间步长累积输入
            mem_potential = mem_potential + input_reshape[t]
            spike = ((mem_potential >= alpha * Vth).float() * Vth).float()  # 脉冲电压
            mem_potential = mem_potential - spike  # 软重置
            spikes.append(spike)
        output = torch.cat(spikes, 0)  # 将spike按照时间维度拼接

        ctx.timesteps = timesteps
        ctx.Vth = Vth
        return output

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():  # 被这句话括起来的部分不会在反向传播中被记录
            input = ctx.saved_tensors[0]
            timesteps = ctx.timesteps
            Vth = ctx.Vth

            input_rate_coding = rate_spikes(input, timesteps)
            grad_output_coding = rate_spikes(grad_output, timesteps) * timesteps

            input_grad = grad_output_coding.clone()
            input_grad[(input_rate_coding < 0) | (input_rate_coding > Vth)] = 0
            input_grad = torch.cat([input_grad for _ in range(timesteps)], 0) / timesteps
            Vth_grad = grad_output_coding.clone()
            Vth_grad[input_rate_coding <= Vth] = 0
            Vth_grad = torch.sum(Vth_grad)
            if torch.cuda.device_count() != 1:
                Vth_grad = sum_tensor(Vth_grad)

            return input_grad, None, Vth_grad, None  # 为什么要返回四个参数？


class IFNeuron(nn.Module):
    def __init__(self, snn_setting):
        super().__init__()
        self.timesteps = snn_setting['timesteps']
        if snn_setting['train_Vth']:
            self.Vth = nn.Parameter(torch.tensor(snn_setting['Vth']))
        else:
            self.Vth = torch.tensor(snn_setting['Vth'])
        self.alpha = snn_setting['alpha']
        self.Vth_bound = snn_setting['Vth_bound']
        self.rate_stat = snn_setting['rate_stat']
        if self.rate_stat:
            self.firing_rate = RateStatus()

    def forward(self, x):
        with torch.no_grad():
            self.Vth.copy_(F.relu(self.Vth - self.Vth_bound) + self.Vth_bound)
        iffunc = IFFunction.apply
        out = iffunc(x, self.timesteps, self.Vth, self.alpha)
        if not self.training and self.rate_stat:
            with torch.no_grad():
                self.firing_rate.append(rate_spikes(out, self.timesteps) / self.Vth)
        return out


class RateStatus(nn.Module):
    '''
    Record the average firing rate of one neuron layer.
    '''
    def __init__(self, max_num=1e6):
        super().__init__()
        self.pool = []
        self.num = 0
        self.max_num = max_num

    def append(self, data):
        self.pool.append(data.view(-1))
        self.num += self.pool[-1].size()[0]
        if self.num > self.max_num:
            self.random_shrink()

    def random_shrink(self):
        tensor = torch.cat(self.pool, 0)
        tensor = tensor[torch.randint(len(tensor), size=[int(self.max_num // 2)])]
        self.pool.clear()
        self.pool.append(tensor)

    def avg(self, max_num=1e6):
        tensor = torch.cat(self.pool, 0)
        if len(tensor) > max_num:
            tensor = tensor[torch.randint(len(tensor), size=[int(max_num)])]
        return tensor.mean()