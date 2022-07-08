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
root_dir = env.SERVER_DOCKER['waymo']
raw_dir = root_dir + 'raw/training/'
val_raw_dir = root_dir + 'raw/validation/'
processed_dir = root_dir + 'processed/interactive/training/'
val_processed_dir = root_dir + 'processed/interactive/validation/'

file_names = [f for f in os.listdir(raw_dir)]
val_file_names = [f for f in os.listdir(val_raw_dir)]

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

if args.resume:
    ### Load previous weight
    weight = os.path.join(config['save_dir'], args.resume)
    state_dict = torch.load(weight)
    
    net.load_state_dict(state_dict)
    CHECKPOINT = int(args.resume[:-5])
else:
    state_dict = net.state_dict()
    ### Initialize partial weight from pretrain weight(Relation Predictor)
    pretrain_weight = './weights/stage_1/14.ckpt'
    pretrain_state_dict = torch.load(pretrain_weight)
    state_dict = model.load_partial_weight_from_pretrain(args, pretrain_state_dict, state_dict)
    
    ### Initialize partial weight from pretrain weight(rename layer name)
    #pretrain_weight = './weights/stage_2/57.ckpt'
    #pretrain_state_dict = torch.load(pretrain_weight)
    #state_dict = model.update_weight(args, pretrain_state_dict, state_dict)
    #torch.save(net.state_dict(), f'{save_dir}/rename.ckpt')
    net.load_state_dict(state_dict)

### Prepare for model
net.train()
epochs = range(CHECKPOINT, EPOCHS)

def train_waymo(logger):
    for epoch in epochs:
        running_total_loss = 0.0
        running_ade = 0.0
        steps = 0
        correct = 0
        correct_size = 0
        file_iter = tqdm(file_names)
        for i, file in enumerate(file_iter):
            file_idx = file[-14:-9]
            raw_path = raw_dir+file
            dataset = Dataset(raw_path, config, processed_dir+f'{file_idx}')
            dataloader = DataLoader(dataset, batch_size=BATCHSIZE, collate_fn=my_collate, num_workers=8)
            dataiter = iter(dataloader)
            for data in dataiter:
                if(data==None):
                    continue
                opt.zero_grad()
                
                pred_class, pred_a, pred_b = net(data)
                loss = loss_fn(args, data, pred_class, pred_a, pred_b)
                loss['Loss'].backward()
                opt.step()
                
                # Logger 
                relation_class = data['relation']
                relation_class_tensor = torch.as_tensor(relation_class).to(device) 
                conf, index = pred_class.max(-1)
                
                correct_tensor = torch.eq(relation_class_tensor, index)
                correct_count = torch.sum((correct_tensor==True).int())
                correct += correct_count.item()
                correct_size += correct_tensor.shape[-1]

                running_total_loss += loss['Loss'].detach().cpu().numpy()
                running_ade += loss['ADE'].detach().cpu().numpy()[-1]
                steps += 1
            file_iter.set_description(f'Epoch: {epoch+1}, Total_Loss: {running_total_loss/steps}, Relation_Accuracy: {correct/correct_size}, ADE: {running_ade/steps}')
            logger.add_scalar('Loss', running_total_loss/steps, epoch*len(file_names) + i) 
            logger.add_scalar('Relation_Acc', correct/steps, epoch*len(file_names) + i) 
            logger.add_scalar('ADE', running_ade/steps, epoch*len(file_names) + i) 
        torch.save(net.state_dict(), f'{save_dir}/{epoch+1}.ckpt')

def val_waymo():
    pass

def main():
    if config['dataset'] == 'waymo':
        log_dir = "logs/joint/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger = SummaryWriter(log_dir)
        train_waymo(logger)
        logger.close()

if __name__ == '__main__':
    main()
