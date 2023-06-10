from torch import nn,square
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1,23), stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.ELU(),
            nn.Dropout(0.38),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=30, kernel_size=(22,1), stride=1, padding=0),
            nn.BatchNorm2d(30),
            nn.ELU(),
            nn.Dropout(0.38),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1,17), stride=1, padding=0),
            nn.BatchNorm2d(30),
            nn.ELU(),
            nn.Dropout(0.38),
        )
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,6), stride=6,padding=0),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1,7), stride=1, padding=0),
            nn.BatchNorm2d(30),
            nn.ELU(),
            nn.Dropout(0.38),
        )
        self.maxpool2 = nn.Sequential(
           nn.MaxPool2d(kernel_size=(1,6), stride=6,padding=0),
        )
        self.grl = GRL_Layer()   

        self.getfeature = nn.Sequential(
            nn.Flatten()
        )

        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(540),
            nn.Linear(540,4),
            nn.Dropout(0.2),
        )
        self.adv = nn.Sequential(
            nn.Flatten(),

            nn.Linear(768,9),
            nn.Dropout(0.2),
        )
        self.out = nn.Sequential(
            nn.Softmax(dim=0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data,std=0.01)
                m.bias.data.fill_(0.01)
                # nn.init.xavier_normal(m.weight.data)
                # nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight.data,std=0.01)
                m.bias.data.fill_(0.01)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool1(out)
        out = self.conv4(out)
        out = self.maxpool2(out)
        out = self.cls(out)
        out = self.out(out)
        return out
    def grl_forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool1(out)
        out = self.conv4(out)
        out = self.maxpool2(out)
        out = self.grl(out)
        out = self.adv(out)
        out = self.out(out)
        return out