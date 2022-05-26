import torch
import math
from torch import nn
import torch.nn.functional as F
import os
from data import WaymoInteractiveDataset, my_collate_fn

# GPU utilization
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create Weights Dir
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.abspath(os.path.join(file_path, os.pardir)))
model_name = os.path.basename(file_path).split(".")[0]
config = {
'epochs': 80,
'observed': 11,
'total': 91,
'batch_size': 4,
'author':'Hong, Kai-Yin',
'account_name':'kaiyin0208.ee07@nycu.edu.tw',
'unique_method_name':'SDC-Centric Multiple Tragets Joint Prediction',
'dataset':'waymo',
}
if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "weights", model_name
    )

# top wrapper of the structure
class Net(nn.Module):
    def __init__(self, config):
        super(Net , self).__init__()
        
        # config parameter
        self.in_dim = config['observed']*3
        self.hidden_dim = 128
        self.out_dim = (config['total']-config['observed'])*2
        map_in_dim = 4
        self.relation_out_dim = 3 #(Pass, Yeild, None)
        
        # initiate encoder
        self.mlp = MLP(self.in_dim, self.hidden_dim, self.hidden_dim)
        self.map_net = MapNet(map_in_dim, self.hidden_dim, self.out_dim)

        # initiate relation_selector
        self.att_lane_a = MultiHeadAttention(self.hidden_dim, 8)
        self.att_lane_b = MultiHeadAttention(self.hidden_dim, 8)

        # initiate relation_predictor
        self.relation_pred = RelationPredictor(self.hidden_dim, self.relation_out_dim)

        # initiate decoder
        #self.marg_pred = MLP(self.hidden_dim, self.hidden_dim, self.out_dim)
        #self.cond_pred = MLP(self.hidden_dim*2, self.hidden_dim, self.out_dim)
    
    def forward(self, data):
        # agent motion encoder
        x_a = data['x_a'].reshape(-1, self.in_dim).to(device)
        x_b = data['x_b'].reshape(-1, self.in_dim).to(device)
        x_a = self.mlp(x_a)
        x_b = self.mlp(x_b)
        
        # lane geometric encoder
        lane_graph = data['lane_graph']
        lane_feature = self.map_net(lane_graph)

        # agents to lane attention
        x_a = x_a.unsqueeze(0)
        x_b = x_b.unsqueeze(0)
        lane_feature  = lane_feature.unsqueeze(0)
        x_a = self.att_lane_a(x_a, lane_feature, lane_feature)
        x_b = self.att_lane_b(x_b, lane_feature, lane_feature)
        
        # relation predictor
        relation = self.relation_pred(x_a, x_b).reshape(1,3)
       
        return relation
        #pass_score = relation[0]
        #yeild_score = relation[1]
        #none_score = relation[2]

        # marginal prediction
        #if none_score > pass_score and none_score > yeild_score:
        #    pred_a = self.marg_pred(x_a)
        #    pred_b = self.marg_pred(x_b)
        
        # joint prediction
        #elif pass_score > yeild_score:
        #    pred_a = self.marg_pred(x_a)
        #    concat = torch.cat([pred_a, x_b])
        #    pred_b = self.cond_pred(concat)
        #else:
        #    pred_b = self.marg_pred(x_b)
        #    concat = torch.cat([pred_b, x_a])
        #    pred_a = self.cond_pred(concat)
 
        #return pred_a, pred_b

class RelationPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RelationPredictor, self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(in_dim*2, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Softmax(dim=1)
                )
    def forward(self, agent_a, agent_b):
        concat = torch.cat((agent_a, agent_b), dim=-1)
        x = self.decoder(concat)

        return x

# MultiHead Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def attention(self,q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        self.scores = scores

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    def forward(self, iq, ik, iv, mask=None):

        bs = iq.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(ik).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(iq).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(iv).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        # ResNet structure
        concat = concat + iq

        output = self.out(concat) + concat

        return output

# SubGraph in VectorNet
class SubGraph(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SubGraph, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim//2),
                nn.LayerNorm(out_dim//2),
                nn.ReLU()
                )
    def forward(self, polylines):

        # node encoder
        # [lanes, lane points, 3] => [lanes, lane points, features]
        whole_polys_feature = self.mlp(polylines)

        # permutation invariant aggregator
        kernel_size = whole_polys_feature.shape[1]
        maxpool = nn.MaxPool1d(kernel_size)
        poly_feature = maxpool(whole_polys_feature.transpose(1,2)).transpose(1,2)

        # concatenation
        whole_polys_feature = torch.cat([whole_polys_feature, poly_feature.repeat(1,10,1)], -1)
        return whole_polys_feature

# Map Information Encoder
class MapNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MapNet, self).__init__()
        local_graph=[]
        for i in range(3):
            local_graph.append(SubGraph(in_dim, hidden_dim, hidden_dim))
            in_dim = hidden_dim
        self.local_graph = nn.ModuleList(local_graph)

    def forward(self, lane_graph):
        whole_polys_feature = lane_graph.to(device)

        # Stack three layer of subgraph
        for i in range(3):
            whole_polys_feature = self.local_graph[i](whole_polys_feature)

        # Read out whole lane feature
        # [lanes, lane points, features] => [lanes, features, lanepoints] => [lanes, features, 1] => [lanes, features]
        kernel_size = whole_polys_feature.shape[1]
        maxpool = nn.MaxPool1d(kernel_size)
        poly_feature = maxpool(whole_polys_feature.transpose(1,2)).squeeze()
        return poly_feature

# simple multi layer perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)

def get_model():
    net = Net(config)
    net = net.to(device)
    params = net.parameters()
    opt= torch.optim.Adam(net.parameters(), lr=1e-4)

    return config, WaymoInteractiveDataset, my_collate_fn, net, opt 
