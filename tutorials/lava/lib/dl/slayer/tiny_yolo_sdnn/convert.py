# nosec # noqa
import os
import argparse
from typing import Any, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd


Width = int
Height = int

class PropheseeAutomotiveSmall(obd.dataset.PropheseeAutomotive):
    def __init__(self,
                 root: str = './',
                 delta_t: int = 1, 
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 seq_len: int = 32,
                 events_ratio: float = 0.07,
                 randomize_seq: bool = False,
                 augment_prob: float = 0.0) -> None:
        super().__init__(root=root, delta_t=delta_t, train=train, size=size, 
                         seq_len=seq_len, randomize_seq=randomize_seq, 
                         events_ratio=events_ratio, augment_prob=augment_prob)

    def __len__(self):
        return 45
    
class PropheseeAutomotiveSmallTrain(obd.dataset.PropheseeAutomotive):
    def __init__(self,
                 root: str = './',
                 delta_t: int = 1, 
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 seq_len: int = 32,
                 events_ratio: float = 0.07,
                 randomize_seq: bool = False,
                 augment_prob: float = 0.0) -> None:
        super().__init__(root=root, delta_t=delta_t, train=train, size=size, 
                         seq_len=seq_len, randomize_seq=randomize_seq, 
                         events_ratio=events_ratio, augment_prob=augment_prob)

    def __len__(self):
        return 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=[1], help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',   type=int, default=1,  help='batch size for dataloader')
    parser.add_argument('-verbose', default=False, action='store_true', help='lots of debug printouts')
    # Model
    parser.add_argument('-model', type=str, default='yolo_kp_events', help='network model')
    # Sparsity
    parser.add_argument('-sparsity', action='store_true', default=False, help='enable sparsity loss')
    parser.add_argument('-sp_lam',   type=float, default=0.01, help='sparsity loss mixture ratio')
    parser.add_argument('-sp_rate',  type=float, default=0.01, help='minimum rate for sparsity penalization')
    # Optimizer
    parser.add_argument('-lr',  type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-wd',  type=float, default=1e-5,   help='optimizer weight decay')
    parser.add_argument('-lrf', type=float, default=0.01,   help='learning rate reduction factor for lr scheduler')
    # Network/SDNN
    parser.add_argument('-threshold',  type=float, default=0.1, help='neuron threshold')
    parser.add_argument('-tau_grad',   type=float, default=0.1, help='surrogate gradient time constant')
    parser.add_argument('-scale_grad', type=float, default=0.2, help='surrogate gradient scale')
    parser.add_argument('-clip',       type=float, default=10, help='gradient clipping limit')
    # Network/SDNN
    parser.add_argument('-cuba_threshold',  type=float, default=0.1, help='neuron threshold')
    parser.add_argument('-cuba_current_decay',   type=float, default=1, help='surrogate gradient time constant')
    parser.add_argument('-cuba_voltage_decay', type=float, default=1, help='surrogate gradient scale')
    parser.add_argument('-cuba_tau_grad',       type=float, default=0.1, help='gradient clipping limit')
    parser.add_argument('-cuba_scale_grad',       type=float, default=15, help='gradient clipping limit')
    # Pretrained model
    parser.add_argument('-load', type=str, default='/home/lecampos/elrond91/lava-dl/Trained_yolo_kp_events_02/network.pt', help='pretrained model')
    # Target generation
    parser.add_argument('-tgt_iou_thr', type=float, default=0.5, help='ignore iou threshold in target generation')
    # YOLO loss
    parser.add_argument('-lambda_coord',    type=float, default=1.0, help='YOLO coordinate loss lambda')
    parser.add_argument('-lambda_noobj',    type=float, default=2.0, help='YOLO no-object loss lambda')
    parser.add_argument('-lambda_obj',      type=float, default=2.0, help='YOLO object loss lambda')
    parser.add_argument('-lambda_cls',      type=float, default=4.0, help='YOLO class loss lambda')
    parser.add_argument('-lambda_iou',      type=float, default=2.0, help='YOLO iou loss lambda')
    parser.add_argument('-alpha_iou',       type=float, default=0.8, help='YOLO loss object target iou mixture factor')
    parser.add_argument('-label_smoothing', type=float, default=0.1, help='YOLO class cross entropy label smoothing')
    parser.add_argument('-track_iter',      type=int,  default=1000, help='YOLO loss tracking interval')
    # Experiment
    parser.add_argument('-exp',  type=str, default='',   help='experiment differentiater string')
    parser.add_argument('-seed', type=int, default=None, help='random seed of the experiment')
    # Training
    parser.add_argument('-epoch',  type=int, default=200, help='number of epochs to run')
    parser.add_argument('-warmup', type=int, default=10,  help='number of epochs to warmup')
    # dataset
    parser.add_argument('-dataset',     type=str,   default='PropheseeAutomotive', help='dataset to use [BDD100K, PropheseeAutomotive]')
    parser.add_argument('-subset',      default=False, action='store_true', help='use PropheseeAutomotive12 subset')
    parser.add_argument('-seq_len',  type=int, default=32, help='number of time continous frames')
    parser.add_argument('-delta_t',  type=int, default=1, help='time window for events')
    parser.add_argument('-event_ratio',  type=float, default=0.07, help='filtering bbox')
    parser.add_argument('-path',        type=str,   default='/home/lecampos/data/prophesee', help='dataset path')
    parser.add_argument('-output_dir',  type=str,   default='.', help='directory in which to put log folders')
    parser.add_argument('-num_workers', type=int,   default=1, help='number of dataloader workers')
    parser.add_argument('-aug_prob',    type=float, default=0.2, help='training augmentation probability')
    parser.add_argument('-clamp_max',   type=float, default=5.0, help='exponential clamp in height/width calculation')

    args = parser.parse_args()

    identifier = f'{args.model}_' + args.exp if len(args.exp) > 0 else args.model
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}'.format(args.seed)

    trained_folder = args.output_dir + '/Trained_' + \
        identifier if len(identifier) > 0 else args.output_dir + '/Trained'
    logs_folder = args.output_dir + '/Logs_' + \
        identifier if len(identifier) > 0 else args.output_dir + '/Logs'

    print(trained_folder)
    writer = SummaryWriter(args.output_dir + '/runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    classes_output = {'BDD100K': 11, 'PropheseeAutomotive': 7}

    print('Creating Network')
    if args.model == 'tiny_yolov3_str':
        Network = obd.models.tiny_yolov3_str.Network
    elif args.model == 'yolo_kp':
        Network = obd.models.yolo_kp.Network
    elif args.model == 'tiny_yolov3_str_events':
        Network = obd.models.tiny_yolov3_str_events.Network
    elif args.model == 'yolo_kp_events':
        Network = obd.models.yolo_kp_events.Network    
    else:
        raise RuntimeError(f'Model type {args.model=} not supported!')
    
    if len(args.gpu) == 1:
        net = Network(threshold=args.threshold,
                      tau_grad=args.tau_grad,
                      scale_grad=args.scale_grad,
                      num_classes=classes_output[args.dataset],
                      clamp_max=args.clamp_max,
                      cuba_params={'threshold': args.cuba_threshold,
                                    'current_decay' : args.cuba_current_decay,
                                    'voltage_decay' : args.cuba_voltage_decay,
                                    'tau_grad'      : args.cuba_tau_grad,   
                                    'scale_grad'    : args.cuba_scale_grad}).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(threshold=args.threshold,
                                            tau_grad=args.tau_grad,
                                            scale_grad=args.scale_grad,
                                            num_classes=classes_output[args.dataset],
                                            clamp_max=args.clamp_max).to(device),
                                    device_ids=args.gpu)
        module = net.module

    if args.sparsity:
        sparsity_montior = slayer.loss.SparsityEnforcer(
            max_rate=args.sp_rate, lam=args.sp_lam)
    else:
        sparsity_montior = None

    print('Loading Network')
    if args.load != '':
        saved_model = args.load
        if saved_model in ['slayer', 'lava-dl']:
            saved_model = slayer.obd.models.__path__[0] + '/Trained_' + args.model + '/network.pt'
        print(f'Initializing model from {saved_model}')
        module.load_model(saved_model)

    if args.model == 'tiny_yolov3_str':
       module.init_model((448, 448, 3))
    elif args.model == 'yolo_kp':
        module.init_model((448, 448, 3))
    elif args.model == 'tiny_yolov3_str_events':
        module.init_model((448, 448, 2))
    elif args.model == 'yolo_kp_events':
        module.init_model((448, 448, 2))
    
    module.load_state_dict(torch.load('/home/lecampos/elrond91/lava-dl/Trained_yolo_kp_events_02' + '/network.pt'))
    module.export_hdf5('/home/lecampos/elrond91/lava-dl/Trained_yolo_kp_events_02' + '/network.net')
    
