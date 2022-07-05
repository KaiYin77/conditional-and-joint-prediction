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

### Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
        "--resume", 
        help="resume from a checkpoint", 
        default="", 
        type=str
)
parser.add_argument(
        "--debug",
        help="debug mode",
        action='store_true',
)
args = parser.parse_args()
### Setting data path
root_dir = env.LAB_PC['waymo']
raw_dir = root_dir + 'raw/training/'
val_raw_dir = root_dir + 'raw/validation/'
test_raw_dir = root_dir + 'raw/testing/'
processed_dir = root_dir + 'processed/interactive/training/'
val_processed_dir = root_dir + 'processed/interactive/validation/'
test_processed_dir = root_dir + 'processed/interactive/testing/'

file_names = [f for f in os.listdir(raw_dir)]
val_file_names = [f for f in os.listdir(val_raw_dir)]
test_file_names = [f for f in os.listdir(test_raw_dir)]

os.makedirs(processed_dir, exist_ok=True)

### GPU utilization
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Create the model
model = import_module(f"module.stage_2")
config, Dataset, my_collate, net, opt = model.get_model()
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
if args.resume:
    weight = os.path.join(config['save_dir'], args.resume)
    state_dict = torch.load(weight)
    net.load_state_dict(state_dict)
else:
    print('[ERROR]: No provided .ckpt')

### Prepare for model
net.eval()

def submit_waymo(logger):
    from waymo_open_dataset.protos import motion_submission_pb2, scenario_pb2
    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.account_name = config['account_name']
    submission.unique_method_name = config['unique_method_name']
    authors = submission.authors
    authors.append(config['author'])

    with torch.no_grad():
        for 
def main():
    if config['dataset'] == 'waymo':
        submit_waymo()

if __name__ == '__main__':
    main()

