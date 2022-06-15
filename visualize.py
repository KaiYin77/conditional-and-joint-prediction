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
from scipy import interpolate
from importlib import import_module
from loss import Loss
import env 

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--weight", 
                    help="trained weight",
                    default="", 
                    type=str
        )
parser.add_argument("--file", 
                    help="select file",
                    default="", 
                    type=int
        )
parser.add_argument("--viz_step", 
                    help="skip scenario",
                    default="", 
                    type=int
        )
parser.add_argument(
        "--debug",
        help="debug mode",
        action='store_true',
)
parser.add_argument(
        "--ablation",
        help="ablation mode",
        action='store_true',
)
parser.add_argument(
        "--single",
        help="single mode",
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
if (args.ablation):
    model = import_module(f"module.stage_2_marg_joint")#For Ablation Study
else:
    model = import_module(f"module.stage_2")
config, Dataset, my_collate, net, opt = model.get_model()
loss_fn = Loss(args, config)
BATCHSIZE = 1 #replace config['batch_size'] of 1 for visualization

### Resume training state
CHECKPOINT = 0
EPOCHS = config['epochs']
OBSERVED = config['observed']
TOTAL = config['total']
PREDICT = config['total'] - config['observed']

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
def val_marg_joint_cond_prediction():
    with torch.no_grad():
        val_file_iter = tqdm(val_file_names)
        for i, file in enumerate(val_file_iter):
            if (i!=int(args.file)):
                continue
            file_idx = file[-14:-9]
            val_raw_path = val_raw_dir+file
            dataset = Dataset(val_raw_path, config, val_processed_dir+f'{file_idx}')
            dataloader = DataLoader(dataset, batch_size=BATCHSIZE, collate_fn=my_collate, num_workers=8)
            dataiter = iter(dataloader)
            for j, data in enumerate(dataiter):
                if(data==None):
                    continue

                marg_a, marg_b, pred_class, pred_a, pred_b = net(data)
                loss = loss_fn(args, data, pred_class, pred_a, pred_b)
                
                # Logger 
                relation_class = data['relation']
                relation_class_tensor = torch.as_tensor(relation_class).to(device) 
                if (args.debug): 
                    print('pred_class: ', pred_class)
                    print('gt_class: ', relation_class_tensor)
                    print('pred_a', pred_a) 
                    print('pred_b', pred_b)

                conf, index = pred_class.max(-1)
                if index == relation_class_tensor:
                    #visualize relation
                    if (args.viz):
                        if(j%int(args.viz_step)!=0):
                            continue
                        control = input("Visualize Next? [y/n]: ")
                        if (control == 'n'):
                            exit()
                        else:
                            marg_a = marg_a.reshape(-1, 2)
                            marg_a = marg_a[4:PREDICT:5,:].cpu().detach().numpy()

                            marg_b = marg_b.reshape(-1, 2)
                            marg_b = marg_b[4:PREDICT:5,:].cpu().detach().numpy()

                            pred_a = pred_a.reshape(-1, 2)
                            pred_a = pred_a[4:PREDICT:5,:].cpu().detach().numpy()
                            
                            pred_b = pred_b.reshape(-1, 2)
                            pred_b = pred_b[4:PREDICT:5,:].cpu().detach().numpy()
                            
                            lane_score_a = net.att_lane_a.scores[0,:,0].cpu()
                            lane_score_b = net.att_lane_b.scores[0,:,0].cpu()
                            
                            draw_marg_joint_cond(args, config, data, lane_score_a, lane_score_b, marg_a, marg_b, index, pred_a, pred_b)


def val_joint_prediction():
    with torch.no_grad():
        running_total_loss = 0.0
        running_ade = 0.0
        steps = 0
        correct = 0
        val_file_iter = tqdm(val_file_names)
        for i, file in enumerate(val_file_iter):
            if (i!=int(args.file)):
                continue
            file_idx = file[-14:-9]
            val_raw_path = val_raw_dir+file
            dataset = Dataset(val_raw_path, config, val_processed_dir+f'{file_idx}')
            dataloader = DataLoader(dataset, batch_size=BATCHSIZE, collate_fn=my_collate, num_workers=8)
            dataiter = iter(dataloader)
            for j, data in enumerate(dataiter):
                if(data==None):
                    continue

                pred_class, pred_a, pred_b = net(data)
                loss = loss_fn(args, data, pred_class, pred_a, pred_b)
                
                # Logger 
                relation_class = data['relation']
                relation_class_tensor = torch.as_tensor(relation_class).to(device) 
                if (args.debug): 
                    print('pred_class: ', pred_class)
                    print('gt_class: ', relation_class_tensor)
                    print('pred_a', pred_a) 
                    print('pred_b', pred_b)

                conf, index = pred_class.max(-1)
                if index == relation_class_tensor:
                    correct += 1
                    #visualize relation
                    if (args.viz):
                        if(j%int(args.viz_step)!=0):
                            continue
                        control = input("Visualize Next? [y/n]: ")
                        if (control == 'n'):
                            exit()
                        else:
                            pred_a = pred_a.reshape(-1, 2)
                            pred_a = pred_a[4:PREDICT:5,:].cpu().detach().numpy()
                            
                            pred_b = pred_b.reshape(-1, 2)
                            pred_b = pred_b[4:PREDICT:5,:].cpu().detach().numpy()
                            
                            lane_score_a = net.att_lane_a.scores[0,:,0].cpu()
                            lane_score_b = net.att_lane_b.scores[0,:,0].cpu()
                            if (args.single):
                                draw_single(config, data, lane_score_a, lane_score_b, index, pred_a, pred_b)
                            else:
                                draw_scenario(args, config, data, lane_score_a, lane_score_b, index, pred_a, pred_b)

                running_total_loss += loss['Loss'].item()
                running_ade += loss['ADE'][-1].item() 
                steps += 1
            val_file_iter.set_description(f'Total_Loss: {running_total_loss/steps}, Relation_Accuracy: {correct/steps}, ADE: {running_ade/steps}')


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
                        control = input("Visualize Next? [y/n]: ")
                        if (control == 'n'):
                            exit()
                        else:
                            if (args.single):
                                draw_single(config, data, index)
                            else:
                                draw_scenario(config, data, index)

                running_loss += loss.item()
                steps += 1
            val_file_iter.set_description(f'CE: {running_loss/steps}, Accuracy: {correct/steps}')

def path_smoothing(waypoints):
    #B_spline
    x=[]
    y=[]
    for point in waypoints:
        x.append(point[0])
        y.append(point[1])
    tck, *rest = interpolate.splprep([x,y])
    u = np.linspace(0,1,num=100)
    smooth = interpolate.splev(u, tck)
    return smooth

def draw_scenario(args, config, input_data, lane_score_a, lane_score_b, pred_label, pred_a=None, pred_b=None):
    import matplotlib
    matplotlib.use('Tkagg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    # Prepare data
    x_a_raw = input_data['x_a']
    y_a_raw = input_data['y_a']
    valid_a = input_data['valid_a']
    valid_x_a_index = np.where(valid_a[:config['observed']]==0)
    valid_y_a_index = np.where(valid_a[config['observed']:]==0)
    x_a = np.delete(x_a_raw, valid_x_a_index, 0)
    y_a = np.delete(y_a_raw, valid_y_a_index, 0)
    
    x_b_raw = input_data['x_b']
    y_b_raw = input_data['y_b']
    valid_b = input_data['valid_b']
    valid_x_b_index = np.where(valid_b[:config['observed']]==0)
    valid_y_b_index = np.where(valid_b[config['observed']:]==0)
    x_b = np.delete(x_b_raw, valid_x_b_index, 0)
    y_b = np.delete(y_b_raw, valid_y_b_index, 0)
     
    relation = input_data['relation']  
    
    # Preprocess attention scores
    lane_score_a = lane_score_a.mean(0)
    lane_score_a = lane_score_a * 0.125 / lane_score_a.max() 
    lane_color_a = [(0.4118+4.7*lane_score_a[i].item(), 0.4118-lane_score_a[i].item(), 0.4118-lane_score_a[i].item()) for i in range(lane_score_a.shape[0])] 
    lane_score_b = lane_score_b.mean(0)
    lane_score_b = lane_score_b * 0.125 / lane_score_b.max() 
    lane_color_b = [(0.4118+4.7*lane_score_b[i].item(), 0.4118-lane_score_b[i].item(), 0.4118-lane_score_b[i].item()) for i in range(lane_score_b.shape[0])] 

    if (args.debug):
        print('lane_score_a: ',lane_score_a)
        print('lane_score_b: ',lane_score_b)
    if (args.debug):
        print(valid_a_index)
        print(valid_b_index)

    figure, axes = plt.subplots(1, 4)
    figure.set_figheight(10)
    figure.set_figwidth(20)
    figure.suptitle('File: {}, Scenario id: {}'.format(args.file, input_data['scenario_id']), fontsize=12)

    axes[0].set_title('input:\ncurrent [-1s ~ +0s]', fontsize=11)
    axes[0].set_facecolor('#2b2b2b')
    axes[0].axhline(y=0, ls='--', color='ghostwhite', zorder=-1)
    axes[0].axvline(x=0, ls='--', color='ghostwhite', zorder=-2) 
    
    axes[1].set_title('[Lane Selection]:\ncurrent [+0s ~ +8s]', fontsize=11)
    axes[1].set_facecolor('#2b2b2b')
    axes[1].axhline(y=0, ls='--', color='ghostwhite', zorder=-1)
    axes[1].axvline(x=0, ls='--', color='ghostwhite', zorder=-2) 

    axes[2].set_title('[stage 1] output_relation:\nfuture [+0s ~ +8s]', fontsize=11)
    axes[2].set_facecolor('#2b2b2b')
    axes[2].axhline(y=0, ls='--', color='ghostwhite', zorder=-1)
    axes[2].axvline(x=0, ls='--', color='ghostwhite', zorder=-2)
    
    axes[3].set_title('[stage 2] output_predict_traj:\nfuture [+0s ~ +8s]', fontsize=11)
    axes[3].set_facecolor('#2b2b2b')
    axes[3].axhline(y=0, ls='--', color='ghostwhite', zorder=-1)
    axes[3].axvline(x=0, ls='--', color='ghostwhite', zorder=-2)
    
    # draw map
    for ax in axes.flat:
        #lane[...,i] => i = [x,y,type, state]
        #type => [undefined, freeway, surface_street, bike_lane]
        #state => [unknown, arrow_red, arrow_yellow, arrow_green, red, yellow, gren, flashing_red, flashing_yellow]
        lane = input_data['lane_graph']
        ax.plot(lane[...,0].T, lane[...,1].T, color='dimgray')
    
    # draw map
    lane_polys_a = []
    lane_polys_b = []
    for i in range(len(lane_color_a)):
        lane_poly_a, = axes[1].plot(lane[i,:,0].T, lane[i,:,1].T, color='dimgray', lw=4, alpha=0.4)
        lane_poly_b, = axes[1].plot(lane[i,:,0].T, lane[i,:,1].T, color='dimgray', lw=4, alpha=0.4)
        lane_polys_a.append(lane_poly_a)
        lane_polys_b.append(lane_poly_b)

    # draw attention scores
    for i in range(len(lane_color_a)):
        lane_polys_a[i].set_color(lane_color_a[i])
    for i in range(len(lane_color_b)):
        lane_polys_b[i].set_color(lane_color_b[i])

    # draw interactive trajectory
    # draw input
    axes[0].add_patch(
            patches.Rectangle(
                (-2.5,-1),
                5,
                2,
                edgecolor='seagreen',
                facecolor='seagreen',
                fill=True,
                label='sdc',
                zorder=999,
                ))
    axes[0].arrow(
            0, 0, 10, 0,
            head_width=1.6,
            width=0.4,
            color='seagreen',
            zorder=999,
            )
    axes[0].plot(x_a[:,0].T, x_a[:,1].T, color='tomato', label='history_a')
    axes[0].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='tomato', label='agent_a')
    axes[0].plot(x_b[:,0].T, x_b[:,1].T, color='royalblue', label='history_b')
    axes[0].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='royalblue', label='agent_b')
    
    # draw lane_selection
    axes[1].add_patch(
            patches.Rectangle(
                (-2.5,-1),
                5,
                2,
                edgecolor='seagreen',
                facecolor='seagreen',
                fill=True,
                label='sdc',
                zorder=999,
                ))
    axes[1].arrow(
            0, 0, 10, 0,
            head_width=1.6,
            width=0.4,
            color='seagreen',
            zorder=999,
            )
    axes[1].plot(x_a[:,0].T, x_a[:,1].T, color='tomato', label='history_a')
    axes[1].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='tomato', label='agent_a')
    axes[1].plot(x_b[:,0].T, x_b[:,1].T, color='royalblue', label='history_b')
    axes[1].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='royalblue', label='agent_b')
    
    # draw relation
    axes[2].add_patch(
            patches.Rectangle(
                (-2.5,-1),
                5,
                2,
                edgecolor='seagreen',
                facecolor='seagreen',
                fill=True,
                label='sdc',
                zorder=999,
                ))
    axes[2].arrow(
            0, 0, 10, 0,
            head_width=1.4,
            width=0.4,
            color='seagreen',
            zorder=999,
            )
    axes[2].plot(x_a[:,0].T, x_a[:,1].T, color='tomato', label='history_a')
    axes[2].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='tomato', label='agent_a')
    axes[2].plot(x_b[:,0].T, x_b[:,1].T, color='royalblue', label='history_b')
    axes[2].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='royalblue', label='agent_b')
     
    if pred_label == 0:
        #a pass b
        label = "right-of-way"
        axes[2].annotate(
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
        axes[2].annotate(
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

    elif pred_label == 1:
        # a yield b
        label = "right-of-way"
        axes[2].annotate(
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
        axes[2].arrow(
            x_b[-1,0], x_b[-1,1], x_a[-1,0], x_a[-1,1],
            head_width=1.0,
            width=0.2,
            color='mediumspringgreen',
            zorder=0,
            alpha=0.7
            )
    # draw joint prediction result
    if pred_a is not None and pred_b is not None: 
        axes[3].add_patch(
                patches.Rectangle(
                    (-2.5,-1),
                    5,
                    2,
                    edgecolor='seagreen',
                    facecolor='seagreen',
                    fill=True,
                    label='sdc',
                    zorder=999,
                    ))
        axes[3].arrow(
                0, 0, 10, 0,
                head_width=1.6,
                width=0.4,
                color='seagreen',
                zorder=999,
                )
        
        axes[3].plot(x_a[:,0].T, x_a[:,1].T, color='tomato', label='history_a')
        axes[3].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='tomato', label='agent_a', zorder=999)
        axes[3].plot(y_a[:,0].T, y_a[:,1].T, color='tomato', label='future_a', linewidth=4.5, alpha=0.5)
        #insert origin point
        pred_a = np.insert(pred_a, 0, x_a[-1,:2], axis=0)
        x_smooth, y_smooth = path_smoothing(pred_a) 
        axes[3].plot(x_smooth, y_smooth, '-', color='crimson', label='pred_a', zorder=99, alpha=1)
        axes[3].plot(x_smooth[-1], y_smooth[-1], '>', color='crimson', alpha=0.8)
        
        axes[3].plot(x_b[:,0].T, x_b[:,1].T, color='royalblue', label='history_b')
        axes[3].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='royalblue', label='agent_b', zorder=999)
        axes[3].plot(y_b[:,0].T, y_b[:,1].T, color='royalblue', label='future_b', linewidth=4.5, alpha=0.5)
        
        #insert origin point
        pred_b = np.insert(pred_b, 0, x_b[-1,:2], axis=0)
        x_smooth, y_smooth = path_smoothing(pred_b) 
        axes[3].plot(x_smooth, y_smooth, '-', color='navy', label='pred_b', zorder=99, alpha=1)
        axes[3].plot(x_smooth[-1], y_smooth[-1], '>', color='navy', alpha=0.8)
    
    for ax in axes.flat:
        plt.sca(ax)
        plt.xlim(-50, 40)
        plt.ylim(-50, 40)
        plt.legend()
    plt.show()

def draw_single(config, input_data, lane_score_a, lane_score_b, pred_label, pred_a=None, pred_b=None):
    import matplotlib
    matplotlib.use('Tkagg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    # Prepare data
    x_a_raw = input_data['x_a']
    y_a_raw = input_data['y_a']
    valid_a = input_data['valid_a']
    valid_x_a_index = np.where(valid_a[:config['observed']]==0)
    valid_y_a_index = np.where(valid_a[config['observed']:]==0)
    x_a = np.delete(x_a_raw, valid_x_a_index, 0)
    y_a = np.delete(y_a_raw, valid_y_a_index, 0)
    
    x_b_raw = input_data['x_b']
    y_b_raw = input_data['y_b']
    valid_b = input_data['valid_b']
    valid_x_b_index = np.where(valid_b[:config['observed']]==0)
    valid_y_b_index = np.where(valid_b[config['observed']:]==0)
    x_b = np.delete(x_b_raw, valid_x_b_index, 0)
    y_b = np.delete(y_b_raw, valid_y_b_index, 0)
     
    relation = input_data['relation']  
    
    # Preprocess attention scores
    lane_score_a = lane_score_a.mean(0)
    lane_score_a = lane_score_a * 0.125 / lane_score_a.max() 
    lane_color_a = [(0.4118+4.7*lane_score_a[i].item(), 0.4118-lane_score_a[i].item(), 0.4118-lane_score_a[i].item()) for i in range(lane_score_a.shape[0])] 
    lane_score_b = lane_score_b.mean(0)
    lane_score_b = lane_score_b * 0.125 / lane_score_b.max() 
    lane_color_b = [(0.4118+4.7*lane_score_b[i].item(), 0.4118-lane_score_b[i].item(), 0.4118-lane_score_b[i].item()) for i in range(lane_score_b.shape[0])] 

    figure, axes = plt.subplots(1, 2)
    axes[0].set_title('No Smoothing', fontsize=11)
    axes[0].set_facecolor('#2b2b2b')
    axes[0].axhline(y=0, ls='--', color='ghostwhite', zorder=-1)
    axes[0].axvline(x=0, ls='--', color='ghostwhite', zorder=-2) 
    axes[1].set_title('Smoothing', fontsize=11)
    axes[1].set_facecolor('#2b2b2b')
    axes[1].axhline(y=0, ls='--', color='ghostwhite', zorder=-1)
    axes[1].axvline(x=0, ls='--', color='ghostwhite', zorder=-2) 
    
    lane = input_data['lane_graph']
    for ax in axes.flat:
        ax.plot(lane[...,0].T, lane[...,1].T, color='dimgray')
    
    # draw joint prediction result
    if pred_a is not None and pred_b is not None:
        pred_a = np.insert(pred_a, 0, x_a[-1,:2], axis=0)
        pred_b = np.insert(pred_b, 0, x_b[-1,:2], axis=0)
        for i in range(2):
            axes[i].add_patch(
                    patches.Rectangle(
                        (-2.5,-1),
                        5,
                        2,
                        edgecolor='seagreen',
                        facecolor='seagreen',
                        fill=True,
                        label='sdc',
                        zorder=999,
                        ))
            axes[i].arrow(
                    0, 0, 10, 0,
                    head_width=1.6,
                    width=0.4,
                    color='seagreen',
                    zorder=999,
                    )
            
            axes[i].plot(x_a[:,0].T, x_a[:,1].T, color='tomato', label='history_a')
            axes[i].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='tomato', label='agent_a', zorder=999)
            axes[i].plot(y_a[:,0].T, y_a[:,1].T, color='tomato', label='future_a', linewidth=4.5, alpha=0.5)
            #insert origin point
            if i == 0:
                axes[i].plot(pred_a[:,0].T, pred_a[:,1].T, '-', color='crimson', label='pred_a', zorder=99, alpha=1)
                axes[i].plot(pred_a[-1,0].T, pred_a[-1,1].T, '>', color='crimson', alpha=0.8)
            elif i == 1:
                x_smooth, y_smooth = path_smoothing(pred_a) 
                axes[i].plot(x_smooth, y_smooth, '-', color='crimson', label='pred_a', zorder=99, alpha=1)
                axes[i].plot(x_smooth[-1], y_smooth[-1], '>', color='crimson', alpha=0.8)
            
            axes[i].plot(x_b[:,0].T, x_b[:,1].T, color='royalblue', label='history_b')
            axes[i].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='royalblue', label='agent_b', zorder=999)
            axes[i].plot(y_b[:,0].T, y_b[:,1].T, color='royalblue', label='future_b', linewidth=4.5, alpha=0.5)
            
            #insert origin point
            if i == 0:
                axes[i].plot(pred_b[:,0].T, pred_b[:,1].T, '-', color='royalblue', label='pred_b', zorder=99, alpha=1)
                axes[i].plot(pred_b[-1,0].T, pred_b[-1,1].T, '>', color='royalblue', alpha=0.8)
            elif i == 1:
                x_smooth, y_smooth = path_smoothing(pred_b) 
                axes[i].plot(x_smooth, y_smooth, '-', color='royalblue', label='pred_b', zorder=99, alpha=1)
                axes[i].plot(x_smooth[-1], y_smooth[-1], '>', color='royalblue', alpha=0.8)
    plt.show()

def draw_marg_joint_cond(args, config, input_data, lane_score_a, lane_score_b, marg_a, marg_b, pred_label, pred_a=None, pred_b=None):
    import matplotlib
    matplotlib.use('Tkagg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    # Prepare data
    x_a_raw = input_data['x_a']
    y_a_raw = input_data['y_a']
    valid_a = input_data['valid_a']
    valid_x_a_index = np.where(valid_a[:config['observed']]==0)
    valid_y_a_index = np.where(valid_a[config['observed']:]==0)
    x_a = np.delete(x_a_raw, valid_x_a_index, 0)
    y_a = np.delete(y_a_raw, valid_y_a_index, 0)
    
    x_b_raw = input_data['x_b']
    y_b_raw = input_data['y_b']
    valid_b = input_data['valid_b']
    valid_x_b_index = np.where(valid_b[:config['observed']]==0)
    valid_y_b_index = np.where(valid_b[config['observed']:]==0)
    x_b = np.delete(x_b_raw, valid_x_b_index, 0)
    y_b = np.delete(y_b_raw, valid_y_b_index, 0)
     
    relation = input_data['relation']  
    
    # Preprocess attention scores
    lane_score_a = lane_score_a.mean(0)
    lane_score_a = lane_score_a * 0.125 / lane_score_a.max() 
    lane_color_a = [(0.4118+4.7*lane_score_a[i].item(), 0.4118-lane_score_a[i].item(), 0.4118-lane_score_a[i].item()) for i in range(lane_score_a.shape[0])] 
    lane_score_b = lane_score_b.mean(0)
    lane_score_b = lane_score_b * 0.125 / lane_score_b.max() 
    lane_color_b = [(0.4118+4.7*lane_score_b[i].item(), 0.4118-lane_score_b[i].item(), 0.4118-lane_score_b[i].item()) for i in range(lane_score_b.shape[0])] 

    if (args.debug):
        print('lane_score_a: ',lane_score_a)
        print('lane_score_b: ',lane_score_b)
    if (args.debug):
        print(valid_a_index)
        print(valid_b_index)

    figure, axes = plt.subplots(1, 2)
    figure.set_figheight(10)
    figure.set_figwidth(20)
    figure.suptitle('File: {}, Scenario id: {}'.format(args.file, input_data['scenario_id']), fontsize=12)

    axes[0].set_title('Marginal:\ncurrent [+0s ~ +8s]', fontsize=11)
    axes[0].set_facecolor('#2b2b2b')
    axes[0].axhline(y=0, ls='--', color='ghostwhite', zorder=-1)
    axes[0].axvline(x=0, ls='--', color='ghostwhite', zorder=-2) 
    
    axes[1].set_title('Joint & Conditional:\ncurrent [+0s ~ +8s]', fontsize=11)
    axes[1].set_facecolor('#2b2b2b')
    axes[1].axhline(y=0, ls='--', color='ghostwhite', zorder=-1)
    axes[1].axvline(x=0, ls='--', color='ghostwhite', zorder=-2) 

    # draw map
    for ax in axes.flat:
        #lane[...,i] => i = [x,y,type, state]
        #type => [undefined, freeway, surface_street, bike_lane]
        #state => [unknown, arrow_red, arrow_yellow, arrow_green, red, yellow, gren, flashing_red, flashing_yellow]
        lane = input_data['lane_graph']
        ax.plot(lane[...,0].T, lane[...,1].T, color='dimgray')
    
    # draw marginal prediction result
    if marg_a is not None and marg_b is not None: 
        axes[0].add_patch(
                patches.Rectangle(
                    (-2.5,-1),
                    5,
                    2,
                    edgecolor='seagreen',
                    facecolor='seagreen',
                    fill=True,
                    label='sdc',
                    zorder=999,
                    ))

        axes[0].arrow(
                0, 0, 10, 0,
                head_width=1.6,
                width=0.4,
                color='seagreen',
                zorder=999,
                )
        
        axes[0].plot(x_a[:,0].T, x_a[:,1].T, color='tomato', label='history_a')
        axes[0].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='tomato', label='agent_a', zorder=999)
        axes[0].plot(y_a[:,0].T, y_a[:,1].T, color='tomato', label='future_a', linewidth=4.5, alpha=0.5)
        #insert origin point
        marg_a = np.insert(marg_a, 0, x_a[-1,:2], axis=0)
        x_smooth, y_smooth = path_smoothing(marg_a) 
        axes[0].plot(x_smooth, y_smooth, '-', color='crimson', label='marg_a', zorder=99, alpha=1)
        axes[0].plot(x_smooth[-1], y_smooth[-1], '>', color='crimson', alpha=0.8)
        
        axes[0].plot(x_b[:,0].T, x_b[:,1].T, color='royalblue', label='history_b')
        axes[0].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='royalblue', label='agent_b', zorder=999)
        axes[0].plot(y_b[:,0].T, y_b[:,1].T, color='royalblue', label='future_b', linewidth=4.5, alpha=0.5) 
        #insert origin point
        marg_b = np.insert(marg_b, 0, x_b[-1,:2], axis=0)
        x_smooth, y_smooth = path_smoothing(marg_b) 
        axes[0].plot(x_smooth, y_smooth, '-', color='navy', label='marg_b', zorder=99, alpha=1)
        axes[0].plot(x_smooth[-1], y_smooth[-1], '>', color='navy', alpha=0.8)

    # draw joint prediction result
    if pred_a is not None and pred_b is not None: 
        axes[1].add_patch(
                patches.Rectangle(
                    (-2.5,-1),
                    5,
                    2,
                    edgecolor='seagreen',
                    facecolor='seagreen',
                    fill=True,
                    label='sdc',
                    zorder=999,
                    ))

        axes[1].arrow(
                0, 0, 10, 0,
                head_width=1.6,
                width=0.4,
                color='seagreen',
                zorder=999,
                )
        
        axes[1].plot(x_a[:,0].T, x_a[:,1].T, color='tomato', label='history_a')
        axes[1].plot(x_a[-1,0].T, x_a[-1,1].T, 'o', color='tomato', label='agent_a', zorder=999)
        axes[1].plot(y_a[:,0].T, y_a[:,1].T, color='tomato', label='future_a', linewidth=4.5, alpha=0.5)
        #insert origin point
        pred_a = np.insert(pred_a, 0, x_a[-1,:2], axis=0)
        x_smooth, y_smooth = path_smoothing(pred_a) 
        axes[1].plot(x_smooth, y_smooth, '-', color='crimson', label='pred_a', zorder=99, alpha=1)
        axes[1].plot(x_smooth[-1], y_smooth[-1], '>', color='crimson', alpha=0.8)
        
        axes[1].plot(x_b[:,0].T, x_b[:,1].T, color='royalblue', label='history_b')
        axes[1].plot(x_b[-1,0].T, x_b[-1,1].T, 'o', color='royalblue', label='agent_b', zorder=999)
        axes[1].plot(y_b[:,0].T, y_b[:,1].T, color='royalblue', label='future_b', linewidth=4.5, alpha=0.5)
        
        #insert origin point
        pred_b = np.insert(pred_b, 0, x_b[-1,:2], axis=0)
        x_smooth, y_smooth = path_smoothing(pred_b) 
        axes[1].plot(x_smooth, y_smooth, '-', color='navy', label='pred_b', zorder=99, alpha=1)
        axes[1].plot(x_smooth[-1], y_smooth[-1], '>', color='navy', alpha=0.8)
    
    for ax in axes.flat:
        plt.sca(ax)
        plt.xlim(-50, 40)
        plt.ylim(-50, 40)
        plt.legend()
    plt.show()

if __name__ == '__main__':
    #val_relation()
    if (args.ablation):
        val_marg_joint_cond_prediction()
    else:
        val_joint_prediction()


