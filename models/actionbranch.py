
import torch
from torch import nn
class  ActionClassificationBranch(nn.Module):
    #To predict action label
    def __init__(self, num_actions, action_feature_dim):        
        super(ActionClassificationBranch, self).__init__()
        self.num_actions = num_actions
        self.action_feature_dim = action_feature_dim
        self.classifier = nn.Linear(self.action_feature_dim, self.num_actions)
        

    def forward(self, action_feature):        
        out={}
        reg_out = self.classifier(action_feature)
        _,indices=torch.max(reg_out,1)
        out['reg_outs']=reg_out
        out['pred_labels']=indices
        out['reg_possibilities']=nn.functional.softmax(reg_out,dim=1)
        return out
