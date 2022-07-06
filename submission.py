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
from data_val_test import WaymoInteractiveDataset, my_collate_fn 
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
args = parser.parse_args()
if args.split != "val" and args.split != "test":
    raise Warning("Please assign data split: [val/test]!!")

### Setting data path
root_dir = env.LAB_PC['waymo']
raw_dir = root_dir + 'raw/training/'
val_raw_dir = root_dir + 'raw/validation_interactive/'
test_raw_dir = root_dir + 'raw/testing_interactive/'
processed_dir = root_dir + 'processed/interactive/training/'
val_processed_dir = root_dir + 'processed/interactive/validation/'
test_processed_dir = root_dir + 'processed/interactive/testing/'

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
model = import_module(f"module.stage_2")
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
                pred_class, pred_a, pred_b = net(data)
                # De-normalize
                rot = data['rot'].to(device)
                orig = data['orig'].to(device)
                pred_a = pred_a.reshape(-1, 2)
                pred_a = torch.matmul(pred_a, torch.inverse(rot))+orig
                pred_a = pred_a[4:80:5]
                pred_b = pred_b.reshape(-1, 2)
                pred_b = torch.matmul(pred_b, torch.inverse(rot))+orig
                pred_b = pred_b[4:80:5]
                # Packing
                predict = submission.scenario_predictions.add()
                predict.scenario_id = data['scenario_id'][0]
                add_joint_predicted_trajectory(predict.joint_prediction.joint_trajectories.add(), data, pred_a, pred_b)#only propose one joint prediction
                
    f = open(f'submit/submit_{target_split}.pb', "wb")
    f.write(submission.SerializeToString())
    f.close()

def add_joint_predicted_trajectory(joint_trajectories, data, pred_a, pred_b):
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
    joint_trajectories.confidence = 1

def main():
    if config['dataset'] == 'waymo':
        '''
        {val, test}
        '''
        submit_waymo(target_split=args.split)

if __name__ == '__main__':
    main()

