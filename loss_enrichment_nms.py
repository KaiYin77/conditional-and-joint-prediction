import torch; torch.autograd.set_detect_anomaly(True)
from torch import nn
from torch.distributions import Normal
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
        self.args = args
        if(self.args.debug):
            print('-------------------')
            print('Observe len: ', self.OBSERVED)
            print('Predict len: ', self.TOTAL- self.OBSERVED)
            print('-------------------')
        self.scene_loss = nn.CrossEntropyLoss()

    def timestamp_stack(self, func, mu, sigma, y, valid):
        p_3 = [mu, sigma, y, valid, 3]
        p_5 = [mu, sigma, y, valid, 5]
        p_8 = [mu, sigma, y, valid, 8]
        loss_3, min_index = func(*p_3)
        loss_5, min_index = func(*p_5)
        loss_8, min_index = func(*p_8)
        loss = torch.stack([loss_3, loss_5, loss_8]).T.reshape(-1)
        loss = torch.cat([loss, loss.mean().unsqueeze(0)])
        return loss, min_index
    
    def timestack_reshape(self, mu, sigma, y, valid, time):
        #reshape [batch, time*coord] -> [batch, time, coord]
        mu = mu.reshape(-1, 12, self.PREDICT, 2)
        sigma = sigma.reshape(-1, 12, self.PREDICT, 2)
        y = y.reshape(-1, 1, self.PREDICT, 4)
        valid = valid.reshape(-1, self.TOTAL).unsqueeze(1).repeat(1, 12, 1)
        #slicing
        mu = mu[:,:,4:time*10:5,:]
        sigma = sigma[:,:,4:time*10:5,:]
        y = y[:,:,4:time*10:5,:2]
        valid = valid[:,:,self.OBSERVED+4:self.OBSERVED+time*10:5]

        return mu, sigma, y, valid

    def min_gaussian_nll(self, mu, sigma, y, valid, time=8):
        mu, sigma, y, valid = self.timestack_reshape(mu, sigma, y, valid, time)

        dist = Normal(loc=mu, scale=sigma)
        loss = -1. * dist.log_prob(y)
        #[batch, candidate, time, coord] -> [batch, candidate, time]
        loss = loss.sum(-1)
        
        if isinstance(valid, torch.Tensor):
            loss[valid==0]=0
        
        #[batch, candidate, time] -> [batch, candidate]
        loss = loss.sum(-1)/(valid.sum(-1) + 1e-10)
        loss[valid.sum(-1)==0] = 0
        
        #[batch, candidate] -> [batch]
        loss, arg_min = loss.min(-1)

        #[batch] -> average loss across batch
        loss = torch.mean(loss)
        return loss, arg_min

    def min_fde(self, mu, sigma, y, valid, time=8):
        mu, sigma, y, valid = self.timestack_reshape(mu, sigma, y, valid, time)
        
        loss = (y-mu)**2
        #[batch, candidate, time, coord] -> [batch, candidate, time]
        loss = loss.sum(-1)
        
        if isinstance(valid, torch.Tensor):
            loss[valid==0]=0
        loss = torch.sqrt(loss + 1e-8)
        
        #[batch, candidate, time] -> [batch, candidate]
        loss = loss[..., -1]
        
        #[batch, candidate] -> [batch]
        loss, arg_min = loss.min(-1)

        #[batch] -> average loss across batch
        loss = torch.mean(loss)
        
        return loss, arg_min

    def min_ade(self, mu, sigma, y, valid, time=8):
        mu, sigma, y, valid = self.timestack_reshape(mu, sigma, y, valid, time)

        loss = (y-mu)**2
        #[batch, candidate, time, coord] -> [batch, candidate, time]
        loss = loss.sum(-1)
        
        if isinstance(valid, torch.Tensor):
            loss[valid==0]=0
        loss = torch.sqrt(loss + 1e-8)
        
        #[batch, candidate, time] -> [batch, candidate]
        loss = loss.sum(-1)/(valid.sum(-1) + 1e-10)
        loss[valid.sum(-1)==0] = 0
        
        #[batch, candidate] -> [batch]
        loss, arg_min = loss.min(-1)

        #[batch] -> average loss across batch
        loss = torch.mean(loss)
        
        return loss, arg_min

    def forward(self, args, data, pred_class, mu_a, sigma_a, mu_b, sigma_b, scene_scores):
        # stage_1 loss
        relation_class = data['relation']
        relation_class_tensor = torch.as_tensor(relation_class).to(device) 
        if (args.debug): 
            print('pred_class: ', pred_class.shape)
            print('gt_class: ', relation_class_tensor.shape)
        relation_loss = self.relation_loss_fn(pred_class, relation_class_tensor)
        
        # stage_2 loss
        ## load agent_a data
        y_a = data['y_a'].to(device)
        valid_a = data['valid_a'].to(device)
        
        a_nll, _ = self.timestamp_stack(self.min_gaussian_nll, mu_a, sigma_a, y_a, valid_a)
        a_ade, _ = self.timestamp_stack(self.min_ade, mu_a, sigma_a, y_a, valid_a)
        a_fde, a_fde_index = self.timestamp_stack(self.min_fde, mu_a, sigma_a, y_a, valid_a)

        ## load agent_b data 
        y_b = data['y_b'].to(device)
        valid_b = data['valid_b'].to(device)
        
        b_nll, _ = self.timestamp_stack(self.min_gaussian_nll, mu_b, sigma_b, y_b, valid_b)
        b_ade, _ = self.timestamp_stack(self.min_ade, mu_b, sigma_b, y_b, valid_b)
        b_fde, b_fde_index = self.timestamp_stack(self.min_fde, mu_b, sigma_b, y_b, valid_b)
        
        ## scene scores
        scene_scores = scene_scores.reshape(-1, 12)
        scene_loss_a = self.scene_loss(scene_scores, a_fde_index) 
        scene_loss_b = self.scene_loss(scene_scores, b_fde_index) 
        # total loss: High-level scene intention loss + keyframe distance loss + trajectory distribution loss
        loss = (
            relation_loss + 
            (scene_loss_a + scene_loss_b)/2 + 
            (a_ade[-1] + b_ade[-1])/2 + 
            (a_fde[-1] + b_fde[-1])/2 + 
            (a_nll[-1] + b_nll[-1])/2
        )
        
        loss_dict = {
            'Loss':loss,
            'rel_ce':relation_loss,
            'scene_ce':(scene_loss_a+scene_loss_b)/2,
            'minNLL':(a_nll+b_nll)/2,
            'minADE':(a_ade+b_ade)/2,
            'minFDE':(a_fde+b_fde)/2,
        }
        return loss_dict
