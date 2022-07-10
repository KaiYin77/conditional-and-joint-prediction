import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import argparse
import datetime

import torch; torch.autograd.set_detect_anomaly(True)
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from tqdm import tqdm
from importlib import import_module
from loss import Loss
import env
from preprocess.data_enrichment_val_test import WaymoInteractiveDataset, my_collate_fn 
### Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
        "--weight", 
        help="trained weight", 
        default="", 
        type=str
)
parser.add_argument(
        "--debug",
        help="debug mode",
        action='store_true',
)
parser.add_argument(
        "--split",
        help="val/test",
        default="",
        type=str,
)
parser.add_argument(
        "--data_on",
        help="lab/docker",
        default="lab",
        type=str,
)
args = parser.parse_args()
if args.split != "val" and args.split != "test":
    raise Warning("Please assign data split: [val/test]!!")

### Setting data path
root_dir = env.WAYMO_ROOT[args.data_on]
raw_dir = root_dir + 'raw/training/'
val_raw_dir = root_dir + 'raw/validation_interactive/'
test_raw_dir = root_dir + 'raw/testing_interactive/'
processed_dir = root_dir + 'processed/interactive_enrichment/training/'
val_processed_dir = root_dir + 'processed/interactive_enrichment/validation/'
test_processed_dir = root_dir + 'processed/interactive_enrichment/testing/'

file_names = [f for f in os.listdir(raw_dir)]
val_file_names = [f for f in os.listdir(val_raw_dir)]
test_file_names = [f for f in os.listdir(test_raw_dir)]

raw_dir_dict = {
    "train": raw_dir,
    "val": val_raw_dir,
    "test": test_raw_dir,
}
file_names_dict = {
    "train": file_names,
    "val": val_file_names,
    "test": test_file_names,
}
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(val_processed_dir, exist_ok=True)
os.makedirs(test_processed_dir, exist_ok=True)
processed_dir_dict = {
    "train": processed_dir,
    "val": val_processed_dir,
    "test": test_processed_dir,
}

### GPU utilization
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Create the model
model = import_module(f"module.stage_2_enrichment_nms")
config, Dataset_train, my_collate, net, opt = model.get_model()
Dataset = WaymoInteractiveDataset #Training & (Val, Test)
loss_fn = Loss(args, config)
BATCHSIZE = config['batch_size']

### Resume training state
if(args.debug):
    print(config)
CHECKPOINT = 0
EPOCHS = config['epochs']
OBSERVED = config['observed']
TOTAL = config['total']

save_dir = config["save_dir"]
os.makedirs(save_dir, exist_ok=True)

### Load previous weight
if args.weight:
    weight = os.path.join(config['save_dir'], args.weight)
    state_dict = torch.load(weight)
    net.load_state_dict(state_dict)
else:
    raise Warning("Please provide .ckpt")

### Prepare for model
net.eval()

def submit_waymo(target_split="val"):
    from waymo_open_dataset.protos import motion_submission_pb2, scenario_pb2
    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.account_name = config['account_name']
    submission.submission_type = 2 #INTERACTION_PREDICTION
    submission.unique_method_name = config['unique_method_name']
    authors = submission.authors
    authors.append(config['author'])
    
    target_file_names = file_names_dict[target_split]
    target_raw_dir = raw_dir_dict[target_split]
    target_processed_dir = processed_dir_dict[target_split]
    with torch.no_grad():
        file_iter = tqdm(target_file_names)
        for file in file_iter:
            file_idx = file[-14:-9]
            target_raw_path = target_raw_dir + file
            dataset = Dataset(target_raw_path, config, target_processed_dir+f'/{file_idx}', data_split=target_split)
            dataloader = DataLoader(dataset, batch_size=1, collate_fn=my_collate)
            dataiter = iter(dataloader)
            for data in dataiter:
                if(data==None):
                    continue
                pred_class, mu_a, sigma_a, mu_b, sigma_b, scene_scores = net(data)
                # De-normalize
                rot = data['rot'].to(device)
                orig = data['orig'].to(device)
                mu_a = mu_a.reshape(12, -1, 2)
                mu_a = torch.matmul(mu_a, torch.inverse(rot))+orig
                mu_a = mu_a[:,4:80:5,:]
                mu_b = mu_b.reshape(12, -1, 2)
                mu_b = torch.matmul(mu_b, torch.inverse(rot))+orig
                mu_b = mu_b[:,4:80:5,:]
                # NMS selection
                scene_scores = scene_scores.reshape(12)
                select_by_nms = non_maxmimum_suppression(data, mu_a, mu_b, scene_scores)
                # Packing
                predict = submission.scenario_predictions.add()
                predict.scenario_id = data['scenario_id'][0]
                # Submit top 6 
                for k in select_by_nms:
                    add_joint_predicted_trajectory(predict.joint_prediction.joint_trajectories.add(), data, mu_a[k], mu_b[k], scene_scores[k])
                
    f = open(f'submit/submit_enrichment_{target_split}.pb', "wb")
    f.write(submission.SerializeToString())
    f.close()

def add_joint_predicted_trajectory(joint_trajectories, data, pred_a, pred_b, scene_score):
    from waymo_open_dataset.protos import motion_submission_pb2
    object_a_trajectory = motion_submission_pb2.ObjectTrajectory()
    object_a_trajectory.object_id = data['id_a'][0] 
    object_a_trajectory.trajectory.center_x[:] = pred_a[:,0]
    object_a_trajectory.trajectory.center_y[:] = pred_a[:,1]
    
    object_b_trajectory = motion_submission_pb2.ObjectTrajectory()
    object_b_trajectory.object_id = data['id_b'][0]
    object_b_trajectory.trajectory.center_x[:] = pred_b[:,0]
    object_b_trajectory.trajectory.center_y[:] = pred_b[:,1]
    
    joint_trajectories.trajectories.append(object_a_trajectory)
    joint_trajectories.trajectories.append(object_b_trajectory)
    joint_trajectories.confidence = scene_score

def non_overlapping_rule():
    pass

def pairwise_distance(input_1, input_2):
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(input_1, input_2)
    return output.item()

def non_diverse_rule(pred_selected, pred):
    fixed_threshold = 2.0
    if pairwise_distance(pred_selected[-1], pred[-1]) < fixed_threshold:
        return True
    return False

def non_maxmimum_suppression(data, pred_a, pred_b, scene_scores):
    scores, index = scene_scores.sort(dim=0, descending=False)
    select_by_nms = []
    candidate_index = index.detach().cpu().numpy().tolist()
    while len(candidate_index) > 0:
        select_by_nms.append(candidate_index.pop())
        selected = select_by_nms[-1]
        for idx in candidate_index:
            if non_diverse_rule(pred_a[selected], pred_a[idx]) and non_diverse_rule(pred_b[selected], pred_b[idx]):
                candidate_index.remove(idx)
            '''
            Could be further develop to decrease overlap rate
            '''

    idxs = index.detach().cpu().numpy()[::-1].tolist()     
    i = 0
    while len(select_by_nms) < 6:
        idx = idxs[i]
        if idx not in select_by_nms:
            select_by_nms.append(idx)
        i += 1
    return select_by_nms
    
def main():
    if config['dataset'] == 'waymo':
        '''
        {val, test}
        '''
        submit_waymo(target_split=args.split)

if __name__ == '__main__':
    main()

