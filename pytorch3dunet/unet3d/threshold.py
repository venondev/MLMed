from matplotlib.cbook import flatten
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb

class ThresholdLayer(nn.Module):
    logged_output=None
    log_output=False

    def __init__(self, in_shape):
        super().__init__()
        num=1
        for x in in_shape:
            num*=x

        self.fc = nn.Linear(num, 2)
        self.sigmoid_function=nn.Sigmoid()

    def forward(self, x):
        sigmoid_param=torch.flatten(x)
        sigmoid_param = self.fc(sigmoid_param)
        b=(self.sigmoid_function(sigmoid_param[0]))*-1
        x=torch.mul(torch.add(x,b),sigmoid_param[1])
        x = self.sigmoid_function(x)
        if ThresholdLayer.log_output:
            ThresholdLayer.logged_output=[b.detach().cpu().numpy(),sigmoid_param[1].detach().cpu().numpy()]

            

        return x