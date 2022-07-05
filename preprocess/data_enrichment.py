import numpy as np
import pandas as pd
from scipy import interpolate
import torch; torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0,'..')
import env

class WaymoInteractiveDataset(Dataset):
    def __init__(self, raw_dir, config, processed_dir, raw=False):
        from waymo_open_dataset.protos import scenario_pb2
        import tensorflow as tf
        if not os.path.isdir(processed_dir) or raw:
            raw_dataset = tf.data.TFRecordDataset(raw_dir)
            self.record = [record.numpy() for record in raw_dataset]
        else:
            raw_dataset = tf.data.TFRecordDataset(raw_dir)
            self.record = [record.numpy() for record in raw_dataset]
            #self.record = [f for f in listdir(processed_dir) if isfile(join(processed_dir, f))]
        
        self.scenario = scenario_pb2.Scenario()
        self.config = config
        self.processed_dir = processed_dir
        self.raw = raw
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def find_closest_distance(self, y_a, valid_a, y_b, valid_b):
        OBSERVED = self.config['observed']
        TOTAL = self.config['total']
        closest_distance = 99999
        t_a = 0
        t_b = 0
        for step_1 in range(TOTAL-OBSERVED):
            if not valid_a[step_1]:
                continue
            for step_2 in range(TOTAL-OBSERVED):
                if not valid_b[step_2]:
                    continue
                distance = (y_a[step_1] - y_b[step_2]) ** 2
                distance = distance.sum(-1)
                distance = torch.sqrt(distance)
                if distance < closest_distance:
                    closest_distance = distance
                    t_a = step_1
                    t_b = step_2
        return closest_distance, t_a, t_b

    def calculate_error_threshold(self, type_a, type_b): 
        object_map = {0:"unset", 1:"vehicle", 2:"pedestrian", 3:"cyclist", 4:"other"}
        if (type_a == 1 or type_b ==1):
            error_threshold = 4
        else:
            error_threshold = 2
        return error_threshold
    
    def moving_average(self, x, n=3) :
        ret = torch.cumsum(x, dim=0)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def smoothing(self, x):
        # Create gaussian kernels
        kernel = Variable(torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]))
        # Apply smoothing
        x_smooth = F.conv1d(x.reshape(1,1,-1), kernel)
        return x_smooth.reshape(-1)
    
    def check_decreasing(self, x):
        smoothing = self.smoothing(x)
        moving_average = self.moving_average(smoothing)
        difference = moving_average[1:] - moving_average[:-1]
        state = (difference < 0)
        sum_of_true = torch.sum(state==True) 
        decision = sum_of_true/state.shape[0] > 0.5
        #if decision:
        #    import matplotlib.pyplot as plt
        #    figure, axes = plt.subplots(1, 4)
        #    axes[0].plot(x)
        #    axes[1].plot(smoothing)
        #    axes[2].plot(moving_average)
        #    axes[3].plot(difference)
        #    plt.show()
        return decision
    def check_increasing(self, x):
        smoothing = self.smoothing(x)
        moving_average = self.moving_average(smoothing)
        difference = moving_average[1:] - moving_average[:-1]
        state = (difference > 0)
        sum_of_true = torch.sum(state==True) 
        decision = (sum_of_true/state.shape[0]) > 0.5
        #print('increasing to zero')
        #print('sum_of_true: ', sum_of_true)
        #print(decision)
        #import matplotlib.pyplot as plt
        #figure, axes = plt.subplots(1, 4)
        #axes[0].plot(x)
        #axes[1].plot(smoothing)
        #axes[2].plot(moving_average)
        #axes[3].plot(difference)
        #plt.show()
        return decision
    def dynamic_static_detection(self, x_a, v_a, x_b, v_b):
        relative_pos = x_a - x_b
        relative_dist = torch.hypot(relative_pos[:,0], relative_pos[:,1]) 
        relative_vel = v_a - v_b
        
        if self.check_decreasing(relative_dist):
            if self.check_increasing(relative_vel):
                relation = 0
            elif self.check_decreasing(relative_vel):
                relation = 1
            else:
                relation = 2
        else:
            relation =2
        return relation

    def downsample(self, polyline, desire_len):
        index = np.linspace(0, len(polyline)-1, desire_len).astype(int)
        return polyline[index]

    def __len__(self):
        return len(self.record)
    
    def __getitem__(self, idx):
        sample_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        try:
            if not self.raw:
                sample = torch.load(sample_path)
            else:
                raise ValueError('raw dataset must be provided')
        except:
            sample = {}
            r = self.record[idx]
            self.scenario.ParseFromString(r)

            OBSERVED = self.config['observed']
            TOTAL = self.config['total']
            
            # Get Index
            sdc_track_index = self.scenario.sdc_track_index
            objects_id_of_interest = self.scenario.objects_of_interest
            # Skip non interactive scenario
            if not objects_id_of_interest:
                return None
            # Skip weird case from waymo datatset 
            if objects_id_of_interest[0] == objects_id_of_interest[1]:
                return None
            
            # SDC
            sdc = self.scenario.tracks[sdc_track_index]
            
            ## sdc_heading
            theta = np.arctan2(sdc.states[OBSERVED-1].velocity_y, sdc.states[OBSERVED-1].velocity_x)
            rot = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            
            ## sdc_traj
            sdc_traj = torch.Tensor([[s.center_x, s.center_y] for s in sdc.states])
            orig = sdc_traj[OBSERVED-1]
            
            # Create Agent Dataset
            interactive_tracks = [t for t in self.scenario.tracks if t.id in objects_id_of_interest] 
            
            agent_a = torch.Tensor([[s.center_x, s.center_y, s.velocity_x, s.velocity_y] for s in interactive_tracks[0].states])
            valid_a = torch.Tensor([s.valid for s in interactive_tracks[0].states])
            valid_a = valid_a[:TOTAL]
            
            agent_b = torch.Tensor([[s.center_x, s.center_y, s.velocity_x, s.velocity_y] for s in interactive_tracks[1].states])
            valid_b = torch.Tensor([s.valid for s in interactive_tracks[1].states])
            valid_b = valid_b[:TOTAL]
            
            ## normalize to sdc
            ### [1] slicing x_a
            x_a = agent_a[:OBSERVED]
            ### translate
            x_a[...,:2] = agent_a[:OBSERVED,:2] - orig
            ### rotate
            x_a[...,:2] = x_a[...,:2].mm(rot)
            x_a[...,2:] = x_a[...,2:].mm(rot)
            ### remove invalid
            x_a[valid_a[:OBSERVED]==0]=0
            
            ### slicing y_a
            y_a = agent_a[OBSERVED:TOTAL]
            ### translate
            y_a[...,:2] = agent_a[OBSERVED:TOTAL,:2] - orig
            ### rotate
            y_a[...,:2] = y_a[...,:2].mm(rot)
            y_a[...,2:] = y_a[...,2:].mm(rot)
            ### remove invalid
            y_a[valid_a[OBSERVED:TOTAL]==0]=0 
            
            ### [2]slicing x_b
            x_b = agent_b[:OBSERVED]
            ### translate
            x_b[...,:2] = agent_b[:OBSERVED,:2] - orig
            ### rotate
            x_b[...,:2] = x_b[...,:2].mm(rot)
            x_b[...,2:] = x_b[...,2:].mm(rot)
            ### remove invalid
            x_b[valid_b[:OBSERVED]==0]=0
            
            ### slicing y_b
            y_b = agent_b[OBSERVED:TOTAL]
            ### translate
            y_b[...,:2] = agent_b[OBSERVED:TOTAL,:2] - orig
            ### rotate
            y_b[...,:2] = y_b[...,:2].mm(rot)
            y_b[...,2:] = y_b[...,2:].mm(rot)
            ### remove invalid
            y_b[valid_a[OBSERVED:TOTAL]==0]=0 
            
            # calculate relation GT 
            closest_distance, t_a, t_b = self.find_closest_distance(y_a[...,:2], valid_a, y_b[...,:2], valid_b)
            error_threshold = self.calculate_error_threshold(interactive_tracks[0].object_type, interactive_tracks[1].object_type)
            if closest_distance <= error_threshold:
                if t_a < t_b:
                   relation = 0 # a pass b 
                else:
                   relation = 1 # a yeild b
            else:
                ## Deal with marginal case 
                v_a = torch.hypot(agent_a[OBSERVED:,2], agent_a[OBSERVED:,3])
                v_b = torch.hypot(agent_b[OBSERVED:,2], agent_b[OBSERVED:,3])
                relation = self.dynamic_static_detection(agent_a[OBSERVED:, :2], v_a, agent_b[OBSERVED:, :2], v_b)
            
            ## concat input with object type [(Timestamp[i])->11, (x, y, heading, vx, vy, object_type)->6]
            x_a = torch.cat([x_a, torch.empty(11,1).fill_(interactive_tracks[0].object_type)], -1)
            x_b = torch.cat([x_b, torch.empty(11,1).fill_(interactive_tracks[1].object_type)], -1)

            # Create Lane Graph Dataset
            lane_graph = []
            dms = self.scenario.dynamic_map_states[OBSERVED-1].lane_states
            state_dict = {l.lane: l.state for l in dms}

            for map_feature in self.scenario.map_features:
                if map_feature.HasField('lane'):
                    id = map_feature.id
                    state = state_dict[id] if id in state_dict else 0
                    lane = map_feature.lane
                    polyline = torch.as_tensor([[feature.x, feature.y, lane.speed_limit_mph, lane.type, state] for feature in lane.polyline])
                    if polyline.shape[0] < 10: continue
                    # fine-grained
                    for i in range(polyline_tensor.shape[0]//10):
                        # normalize to sdc coordinate
                        normalize_polyline = torch.clone(polyline_tensor[i*10:i*10+10])
                        normalize_polyline[:, :2] = normalize_polyline[:, :2] - orig 
                        normalize_polyline[:, :2] = normalize_polyline[:, :2].mm(rot)
                        lane_graph.append(normalize_polyline)
            
            # Stack all list
            sample['sdc'] = sdc_traj

            sample['x_a'] = x_a
            sample['y_a'] = y_a
            sample['valid_a'] = valid_a 

            sample['x_b'] = x_b
            sample['y_b'] = y_b
            sample['valid_b'] = valid_b
            
            sample['relation'] = relation
            
            sample['rot'] = torch.as_tensor(rot) 
            sample['orig'] = torch.as_tensor(orig) 
            
            sample['lane_graph'] = torch.zeros(1,10,5) if not lane_graph else torch.stack(lane_graph)
            sample['scenario_id'] = self.scenario.scenario_id 
            
            #torch.save(sample, sample_path) 
        
        return sample


def my_collate_fn(batch):
    batch = list(filter(lambda sample: sample is not None, batch))
    if len(batch) == 0:
        return None
    '''
    Fetch sample's key
    '''
    elem = [k for k, v in batch[0].items()]
    '''
    Prepare util function 
    '''
    def _collate_util(input_list, key):
        if isinstance(input_list[0], torch.Tensor):
            return torch.cat(input_list, 0)
        return input_list

    def _get_object_type(input_list):
        return input_list[0,-1]

    '''
    Collate sample data to collate
    '''
    collate = {key: _collate_util([d[key] for d in batch], key) for key in elem}
    collate.update({'a_object_type':_get_object_type(collate['x_a'])})
    collate.update({'b_object_type':_get_object_type(collate['x_b'])})
    return collate

def analysis_interactive_data():
    
    ### Setting data path
    root_dir = env.LAB_PC['waymo']
    val_raw_dir = root_dir + 'raw/validation/'
    val_processed_dir = root_dir + 'processed/interactive_enrichment/validation/'
    val_file_names = [f for f in os.listdir(val_raw_dir)]

    os.makedirs(val_processed_dir, exist_ok=True)

    ### GPU utilization
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ### Start
    statistic = {
        0: 0,#pass
        1: 0,#yeild
        2: 0,#unrelated
    }

    Dataset = WaymoInteractiveDataset
    config = {
    'epochs': 80,
    'observed': 11,
    'total': 91,
    'batch_size': 1,
    'author':'Hong, Kai-Yin',
    'account_name':'kaiyin0208.ee07@nycu.edu.tw',
    'unique_method_name':'SDC-Centric Multiple Targets Joint Prediction',
    'dataset':'waymo',
    'stage':'relation_stage',
    }
    file_iter = tqdm(val_file_names)
    for i, file in enumerate(file_iter):
        file_idx = file[-14:-9]
        raw_path = val_raw_dir+file
        dataset = Dataset(raw_path, config, val_processed_dir+f'{file_idx}')
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=my_collate_fn, num_workers=8)
        dataiter = iter(dataloader)
        for data in dataiter:
            if(data==None):
                continue
            relation_class_list = data['relation']
            relation_class = relation_class_list[-1]
            statistic[relation_class] += 1 
    
    print('Total cases: ', statistic)

def main():
    analysis_interactive_data()

if __name__ == '__main__':
    main()
