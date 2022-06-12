import torch; torch.autograd.set_detect_anomaly(True)
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Wrapper of all loss
class Loss(nn.Module):
    def __init__(self, args, config):
        super(Loss, self).__init__()
        self.relation_loss_fn = nn.CrossEntropyLoss()
        self.OBSERVED = config['observed']
        self.TOTAL = config['total']
        self.PREDICT = config['total'] - config['observed']
        if(args.debug):
            print('-------------------')
            print('Observe len: ', self.OBSERVED)
            print('Predict len: ', self.TOTAL- self.OBSERVED)
            print('-------------------')
    
    def ade(self, y, out, valid=None):
        loss = (y-out)**2
        loss = loss.sum(-1)
        if isinstance(valid, torch.Tensor):
            loss[valid==0]=0
        loss = torch.sqrt(loss + 1e-8)
        loss = loss.sum(-1)/(valid.sum(-1) + 1e-10)
        return loss

    def forward(self, args, data, pred_class, pred_a, pred_b):
        # stage_1 loss
        relation_class = data['relation']
        relation_class_tensor = torch.as_tensor(relation_class).to(device) 
        if (args.debug): 
            print('pred_class: ', pred_class)
            print('gt_class: ', relation_class_tensor)
        relation_loss = self.relation_loss_fn(pred_class, relation_class_tensor)
        
        # stage_2 loss
        ## load agent_a data
        gt_a = data['y_a'].to(device)
        valid_a = data['valid_a'].to(device)
        
        gt_a = gt_a[4:self.PREDICT:5,:]
        valid_a = valid_a[self.OBSERVED::5]
        pred_a = pred_a.reshape(-1, 2)
        pred_a = pred_a[4:self.PREDICT:5,:]
        
        a_loss = self.ade(gt_a, pred_a, valid_a)
        
        ## load agent_b data 
        gt_b = data['y_b'].to(device)
        valid_b = data['valid_b'].to(device)
        
        gt_b = gt_b[4:self.PREDICT:5,:]
        valid_b = valid_b[self.OBSERVED::5]
        pred_b = pred_b.reshape(-1, 2)
        pred_b = pred_b[4:self.PREDICT:5,:]
        b_loss = self.ade(gt_b, pred_b, valid_b)
        
        # total loss
        loss = relation_loss + (a_loss + b_loss)/2
        
        loss_dict = {
            'Loss':loss,
            'CE':relation_loss,
            'a_ADE':a_loss,
            'b_ADE':b_loss,
            'ADE':(a_loss+b_loss)/2
        }
        return loss_dict
