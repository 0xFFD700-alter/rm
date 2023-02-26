import math
import warnings
import torch
from torch import nn
from functools import partial
from itertools import repeat
import collections.abc

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_3tuple(bias)
        drop_probs = to_3tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, int(hidden_features * 4), bias=bias[1])
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(drop_probs[1])
        self.fc3 = linear_layer(int(hidden_features * 4), out_features, bias=bias[2])
        self.drop3 = nn.Dropout(drop_probs[2])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.drop3(x)
        return x

class layers_scale_mlp_blocks(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, init_values=1e-4):
        super().__init__()
        self.norm = Affine(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(16.0 * dim), act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    
    def forward(self, x):
        x = x + self.drop_path(self.gamma * self.mlp(self.norm(x)))
        return x

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

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

class DoraNet(nn.Module):
    def __init__(self, num_outputs=4, dim=8, depth=6, drop_rate=0.2,
                 act_layer=nn.GELU, drop_path_rate=0., init_scale=1e-4):
        super().__init__()
        self.num_outputs = num_outputs
        self.dim = dim
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            layers_scale_mlp_blocks(
                dim=dim, drop=drop_rate, drop_path=dpr[i],
                act_layer=act_layer, init_values=init_scale)
            for i in range(depth)])
        self.norm = Affine(dim)
        self.head = nn.Linear(dim, num_outputs)
        self.apply(self._init_weights)
        self.classifier = binary_classifier_mlp(dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
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
        cls = self.classifier(x)
        for i , blk in enumerate(self.blocks):
            x  = blk(x)
        x = self.head(x)
        if self.training:
            return x, cls
        else:
            return (x * (158.7472538974973 - 57.70791516029391) - 158.7472538974973) * (torch.sigmoid(cls) > 0.5).float()

def main():    
    b = 1
    model = DoraNet()
    x = torch.zeros((b, 2))
    pathloss = torch.zeros((b, 4))
    cls = torch.zeros((b, 4))

    model.train()
    p_pathloss, p_cls = model(x)
    print(torch.mean(torch.abs(p_pathloss - pathloss)), torch.mean(torch.abs(p_cls - cls)))

    model.eval()
    p_pathloss = model(x)
    print(torch.mean(torch.abs(p_pathloss - pathloss)))

    # print(list(model.children()))

    for child in model.children():
        print(isinstance(child, nn.LayerNorm))
        
if __name__ == '__main__':
    main()