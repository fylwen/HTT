from torch import nn
from models.utils import act_str2func

class MultiLayerPerceptron(nn.Module):
    def __init__(self, base_neurons=[512, 512, 512], out_dim=8,act_hidden='elu',act_final='elu'):
        super().__init__()

        layers = []
        act_str2func_=act_str2func()
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            if act_hidden!='none':
                layers.append(act_str2func_[act_hidden])

        layers.append(nn.Linear(out_neurons, out_dim))
        if act_final!='none':
            layers.append(act_str2func_[act_final])
       

        self.mlp = nn.Sequential(*layers)

    def forward(self, inp):
        out = self.mlp(inp)
        return out

