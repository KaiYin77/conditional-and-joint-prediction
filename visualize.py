import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.append('..')
import argparse

import tensorflow as tf
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np 
from importlib import import_module
import env 
# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--weight", 
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
        "--viz",
        help="viz mode",
        action='store_true',
)
args = parser.parse_args()

### Setting data path
root_dir = env.LAB_PC['waymo']
raw_dir = root_dir + 'raw/validation/'
val_raw_dir = root_dir + 'raw/validation/'
processed_dir = root_dir + 'processed/interactive/validation/'
val_processed_dir = root_dir + 'processed/interactive/validation/'

file_names = sorted([f for f in os.listdir(raw_dir)])
val_file_names = sorted([f for f in os.listdir(val_raw_dir)])

# GPU utilization
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Create the model
model = import_module(f"model")
config, Dataset, my_collate, net, opt = model.get_model()
BATCHSIZE = 1 #replace config['batch_size'] of 1 for visualization

### Resume training state
CHECKPOINT = 0
EPOCHS = config['epochs']
OBSERVED = config['observed']
TOTAL = config['total']

save_dir = config["save_dir"]
os.makedirs(save_dir, exist_ok=True)

if args.weight:
    ### Load previous weight
    weight = os.path.join(config['save_dir'], args.weight)
    state_dict = torch.load(weight)
    net.load_state_dict(state_dict)
else:
    print('no weight provided...')
### Prepare for model
net.eval()
criterion = nn.CrossEntropyLoss()

def val_relation():
    with torch.no_grad():
        running_loss = 0.0
        steps = 0
        correct = 0
        val_file_iter = tqdm(val_file_names)
        for i, file in enumerate(val_file_iter):
            file_idx = file[-14:-9]
            val_raw_path = val_raw_dir+file
            dataset = Dataset(val_raw_path, config, val_processed_dir+f'{file_idx}')
            dataloader = DataLoader(dataset, batch_size=BATCHSIZE, collate_fn=my_collate, num_workers=8)
            dataiter = iter(dataloader)
            for j, data in enumerate(dataiter):
                if(data==None):
                    continue

                outputs = net(data)
                relation_class = data['relation']
                relation_class_tensor = torch.as_tensor(relation_class).to(device)
                
                if (args.debug): 
                    print('pred_class: ', outputs)
                    print('gt_class: ', relation_class_tensor)

                loss = criterion(outputs, relation_class_tensor)
                conf, index = outputs.max(-1)
                if index == relation_class_tensor:
                    correct += 1
                    #visualize relation
                    if (args.viz): 
                        draw_scenario(config, data, index)

                running_loss += loss.item()
                steps += 1
            val_file_iter.set_description(f'CE: {running_loss/steps}, Accuracy: {correct/steps}')

def draw_scenario(config, input_data, pred_label):
    import matplotlib
    matplotlib.use('Tkagg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    x_a = input_data['x_a']
    y_a = input_data['y_a']
    valid_a = input_data['valid_a']
    x_b = input_data['x_b']
    y_b = input_data['y_b']
    valid_b = input_data['valid_b']
    relation = input_data['relation'] 
    
    figure, axes = plt.subplots(1, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    axes[0].set_title('current')
    axes[1].set_title('future')
    # draw map
    for ax in axes.flat:
        lane = input_data['lane_graph']
        ax.plot(lane[...,0].T, lane[...,1].T, color='gray')
    # draw interactive trajectory
    # draw current
    axes[0].plot(x_a[:,0].T, x_a[:,1].T, color='red', label='history_a')
    axes[0].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='red')
    axes[0].plot(x_b[:,0].T, x_b[:,1].T, color='darkblue', label='history_b')
    axes[0].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='darkblue')
    
    if pred_label == 0:
        #a pass b
        label = "right-of-way"
        axes[0].annotate(
                label, 
                (x_a[-1,0].T, x_a[-1,1].T),
                xycoords='data',
                xytext=(10,-40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.2"),
                ha='center', 
                fontsize=8, 
                color='white',  
                backgroundcolor="black"
                )

    elif pred_label == 1:
        # a yield b
        label = "right-of-way"
        axes[0].annotate(
                label, 
                (x_b[-1,0].T, x_b[-1,1].T),
                xycoords='data',
                xytext=(10,-40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.2"),
                ha='center', 
                fontsize=8, 
                color='white',  
                backgroundcolor="black"
                )
    # draw future 
    axes[1].plot(x_a[:,0].T, x_a[:,1].T, color='red', label='history_a')
    axes[1].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='red')
    axes[1].plot(y_a[:,0].T, y_a[:,1].T, color='red', label='future_a')
    axes[1].plot(y_a[-1,0].T, y_a[-1,1].T, '>', color='red', alpha=0.7)
    
    axes[1].plot(x_b[:,0].T, x_b[:,1].T, color='darkblue', label='history_b')
    axes[1].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='darkblue')
    axes[1].plot(y_b[:,0].T, y_b[:,1].T, color='darkblue', label='future_b')
    axes[1].plot(y_b[-1,0].T, y_b[-1,1].T, '>', color='darkblue', alpha=0.7)
    
    plt.show()

if __name__ == '__main__':
    val_relation()


