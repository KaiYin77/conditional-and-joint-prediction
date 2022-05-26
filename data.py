import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import os
from os import listdir
from os.path import isfile, join

class WaymoInteractiveDataset(Dataset):
    def __init__(self, raw_dir, config, processed_dir, raw=False):
        from waymo_open_dataset.protos import scenario_pb2
        import tensorflow as tf
        if not os.path.isdir(processed_dir) or raw:
            raw_dataset = tf.data.TFRecordDataset(raw_dir)
            self.record = [record.numpy() for record in raw_dataset]
        else:
            self.record = [f for f in listdir(processed_dir) if isfile(join(processed_dir, f))]
        self.scenario = scenario_pb2.Scenario()
        self.config = config
        self.processed_dir = processed_dir
        self.raw = raw
        os.makedirs(self.processed_dir, exist_ok=True)
    
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
            # Skip Non interactive scenario
            if not objects_id_of_interest:
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
            agent_a = torch.Tensor([[s.center_x, s.center_y] for s in interactive_tracks[0].states])
            valid_a = torch.Tensor([s.valid for s in interactive_tracks[0].states])
            valid_a = valid_a[:TOTAL]
            
            agent_b = torch.Tensor([[s.center_x, s.center_y] for s in interactive_tracks[1].states])
            valid_b = torch.Tensor([s.valid for s in interactive_tracks[1].states])
            valid_b = valid_b[:TOTAL]
            
            ## normalize to sdc
            x_a = agent_a[:OBSERVED] - orig
            y_a = agent_a[OBSERVED:TOTAL] - orig
            x_a = x_a.mm(rot)
            y_a = y_a.mm(rot)
            x_a[valid_a[:OBSERVED]==0]=0
            y_a[valid_a[OBSERVED:]==0]=0 
            
            x_b = agent_b[:OBSERVED] - orig
            y_b = agent_b[OBSERVED:TOTAL] - orig
            x_b = x_b.mm(rot)
            y_b = y_b.mm(rot)
            x_b[valid_b[:OBSERVED]==0]=0
            y_b[valid_b[OBSERVED:]==0]=0 
            
            # Calculate Relation GT 
            #closest_distance, t1, t2 = self.find_closest_distance(y_a, valid_a, y_b, valid_b)
            #if closest_distance <= 2:
            #   if t1 < t2:
            #       relation = 0 
            #   else:
            #       relation = 0
            relation = 2
            ## concat input with object type [(Timestamp[i])->11, (x, y, object_type)->3]
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
                    polyline = torch.as_tensor([[feature.x, feature.y, lane.type, state] for feature in lane.polyline])
                    if polyline.shape[0] < 10: continue
                    # normaliza to sdc coordinate
                    normalize_polyline = torch.clone(polyline)
                    normalize_polyline[:,:2] = normalize_polyline[:,:2] - orig
                    normalize_polyline[:,:2] = normalize_polyline[:,:2].mm(rot)
                    normalize_polyline = self.downsample(normalize_polyline, 10)
                    lane_graph.append(normalize_polyline)
            
            # Stack all list
            sample['x_a'] = x_a
            sample['y_a'] = y_a
            sample['valid_a'] = valid_a 

            sample['x_b'] = x_b
            sample['y_b'] = y_b
            sample['valid_b'] = valid_b
            
            sample['relation'] = relation
            
            sample['rot'] = torch.as_tensor(rot) 
            sample['orig'] = torch.as_tensor(orig) 
            
            sample['lane_graph'] = torch.zeros(1,10,4) if not lane_graph else torch.stack(lane_graph)
            sample['scenario_id'] = self.scenario.scenario_id 
            
            torch.save(sample, sample_path) 
        
        return sample


def my_collate_fn(batch):
    batch = list(filter(lambda sample: sample is not None, batch))
    if len(batch) == 0:
        return default_collate(batch)
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
