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
    
    def timestamp_stack(self, func, out, y, valid):
        p_3 = [out, y, valid, 3]
        p_5 = [out, y, valid, 5]
        p_8 = [out, y, valid, 8]
        loss_3 = func(*p_3)
        loss_5 = func(*p_5)
        loss_8 = func(*p_8)
        loss = torch.stack([loss_3, loss_5, loss_8]).T.reshape(-1)
        loss = torch.cat([loss, loss.mean().unsqueeze(0)])
        return loss
    
    def timestack_reshape(self, out, y, valid, time):
        #reshape
        out = out.reshape(self.PREDICT, 2)
        y = y.reshape(self.PREDICT, 2)
        #slicing
        out = out[4:time*10:5]
        y = y[4:time*10:5]
        valid = valid[self.OBSERVED+4:self.OBSERVED+time*10:5]

        return out, y, valid
        

    def ade(self, out, y, valid, time=8):
        out, y, valid = self.timestack_reshape(out, y, valid, time)

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
        y_a = data['y_a'].to(device)
        valid_a = data['valid_a'].to(device)
        
        a_loss = self.timestamp_stack(self.ade, pred_a, y_a, valid_a)
        
        ## load agent_b data 
        y_b = data['y_b'].to(device)
        valid_b = data['valid_b'].to(device)
        
        b_loss = self.timestamp_stack(self.ade, pred_b, y_b, valid_b)
        
        # total loss
        loss = relation_loss + (a_loss[-1] + b_loss[-1])/2
        
        loss_dict = {
            'Loss':loss,
            'CE':relation_loss,
            'a_ADE':a_loss,
            'b_ADE':b_loss,
            'ADE':(a_loss+b_loss)/2
        }
        return loss_dict
