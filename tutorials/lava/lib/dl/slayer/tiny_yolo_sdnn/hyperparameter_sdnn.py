import numpy as np
import os
from typing import Tuple
import torch
from torch.utils.data import random_split
from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
import math


Width = int
Height = int


class CosineScheduler:
        def __init__(self, max_update, base_lr=0.01, final_lr=0,
                warmup_steps=0, warmup_begin_lr=0):
            self.base_lr_orig = base_lr
            self.max_update = max_update
            self.final_lr = final_lr
            self.warmup_steps = warmup_steps
            self.warmup_begin_lr = warmup_begin_lr
            self.max_steps = self.max_update - self.warmup_steps

        def get_warmup_lr(self, epoch):
            increase = (self.base_lr_orig - self.warmup_begin_lr) \
                        * float(epoch) / float(self.warmup_steps)
            return self.warmup_begin_lr + increase

        def __call__(self, epoch):
            if epoch < self.warmup_steps:
                # print('scheduler ',self.get_warmup_lr(epoch))
                return self.get_warmup_lr(epoch)
            if epoch <= self.max_update:
                self.base_lr = self.final_lr + (
                    self.base_lr_orig - self.final_lr) * (1 + math.cos(
                    math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
            # print('scheduler ', self.base_lr)
            return self.base_lr

class PropheseeAutomotiveSmall(obd.dataset.PropheseeAutomotive):
    def __init__(self,
                 root: str = './',
                 delta_t: int = 1, 
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 seq_len: int = 32,
                 events_ratio: float = 0.07,
                 randomize_seq: bool = False,
                 augment_prob: float = 0.0,
                 reduce_classes: bool = False) -> None:
        super().__init__(root=root, delta_t=delta_t, train=train, size=size, 
                         seq_len=seq_len, randomize_seq=randomize_seq, 
                         events_ratio=events_ratio, augment_prob=augment_prob,
                         reduce_classes=reduce_classes)

    def __len__(self):
        return 50
    
class PropheseeAutomotiveSmallTrain(obd.dataset.PropheseeAutomotive):
    def __init__(self,
                 root: str = './',
                 delta_t: int = 1, 
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 seq_len: int = 32,
                 events_ratio: float = 0.07,
                 randomize_seq: bool = False,
                 augment_prob: float = 0.0,
                 reduce_classes: bool = False) -> None:
        super().__init__(root=root, delta_t=delta_t, train=train, size=size, 
                         seq_len=seq_len, randomize_seq=randomize_seq, 
                         events_ratio=events_ratio, augment_prob=augment_prob,
                         reduce_classes=reduce_classes)

    def __len__(self):
        return 10


def train_events(config):
    
    classes_output = {'BDD100K': 11, 'PropheseeAutomotive': 7}
    
    print('Using GPUs {}'.format(config["gpu"]))
    device = torch.device('cuda:{}'.format(config["gpu"][0]))
    
    print('Creating Network')
    if config["model"] == 'tiny_yolov3_str':
        Network = obd.models.tiny_yolov3_str.Network
    elif config["model"] == 'yolo_kp':
        Network = obd.models.yolo_kp.Network
    elif config["model"] == 'tiny_yolov3_str_events':
        Network = obd.models.tiny_yolov3_str_events.Network
    elif config["model"] == 'yolo_kp_events':
        Network = obd.models.yolo_kp_events.Network    
    else:
        raise RuntimeError(f'Model type {config["model"]=} not supported!')
    
    if len(config["gpu"]) == 1:
        net = Network(threshold = config["threshold"],
                      tau_grad = config["tau_grad"],
                      scale_grad = config["scale_grad"],
                      num_classes = classes_output[config["dataset"]],
                      clamp_max =config["clamp_max"]).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(threshold=config["threshold"],
                                            tau_grad=config["tau_grad"],
                                            scale_grad=config["scale_grad"],
                                            num_classes=classes_output[config["dataset"]],
                                            clamp_max=config["clamp_max"]).to(device),
                                    device_ids=config["gpu"])
        module = net.module
    
    
    if config["sparsity"]:
        sparsity_montior = slayer.loss.SparsityEnforcer(
            max_rate=config["sp_rate"], lam=config["sp_lam"])
    else:
        sparsity_montior = None
    
    if config["model"] == 'tiny_yolov3_str':
       module.init_model((448, 448, 3))
    elif config["model"] == 'yolo_kp':
        module.init_model((448, 448, 3))
    elif config["model"] == 'tiny_yolov3_str_events':
        module.init_model((448, 448, 2))
    elif config["model"] == 'yolo_kp_events':
        module.init_model((448, 448, 2))

    # Define optimizer module.
    print('Creating Optimizer')
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=config["lr"],
                                 weight_decay=config["wd"])

    # Define learning rate scheduler
    # def lf(x):
    #     return (min(x / config["warmup"], 1)
    #             * ((1 + np.cos(x * np.pi / config["epoch"])) / 2)
    #             * (1 - config["lrf"])
    #             + config["lrf"])

    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    scheduler = CosineScheduler(config["epoch"], 
                                warmup_steps=config["warmup"], 
                                base_lr=config["lr"], 
                                final_lr=config["lr"]*0.1)
    
    yolo_target = obd.YOLOtarget(anchors=net.anchors,
                                 scales=net.scale,
                                 num_classes=net.num_classes,
                                 ignore_iou_thres=config["tgt_iou_thr"])

    # To restore a checkpoint, use `train.get_checkpoint()`.
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    print('Creating Dataset')

    if config["dataset"] == 'BDD100K':
        train_set = obd.dataset.BDD(root=config["path"], dataset='track',
                                    train=True, augment_prob=config["aug_prob"],
                                    randomize_seq=True)
        test_set = obd.dataset.BDD(root=config["path"], dataset='track',
                                   train=False, randomize_seq=True)
        
        train_loader = DataLoader(train_set,
                                  batch_size=config["b"],
                                  shuffle=True,
                                  collate_fn=yolo_target.collate_fn,
                                  num_workers=config["num_workers"],
                                  pin_memory=True)
        test_loader = DataLoader(test_set,
                                 batch_size=config["b"],
                                 shuffle=True,
                                 collate_fn=yolo_target.collate_fn,
                                 num_workers=config["num_workers"],
                                 pin_memory=True)        
    elif config["dataset"] == 'PropheseeAutomotive':
        if config["subset"]:
            train_set = PropheseeAutomotiveSmall(root=config["path"], train=True, 
                                                augment_prob=config["aug_prob"], 
                                                randomize_seq=True,
                                                events_ratio = config["event_ratio"],
                                                delta_t=config["delta_t"],
                                                seq_len=config["seq_len"],
                                                reduce_classes=True)
            
            test_set = PropheseeAutomotiveSmallTrain(root=config["path"], train=False, 
                                                augment_prob=config["aug_prob"], 
                                                randomize_seq=True,
                                                events_ratio = config["event_ratio"],
                                                delta_t=config["delta_t"],
                                                seq_len=config["seq_len"],
                                                reduce_classes=True)
            print('Using PropheseeAutomotiveSmall Dataset')
        else:      
            train_set = obd.dataset.PropheseeAutomotive(root=config["path"], train=True, 
                                                augment_prob=config["aug_prob"], 
                                                randomize_seq=True,
                                                events_ratio = config["event_ratio"],
                                                delta_t=config["delta_t"],
                                                seq_len=config["seq_len"])
            
            test_set = obd.dataset.PropheseeAutomotive(root=config["path"], train=False, 
                                                augment_prob=config["aug_prob"], 
                                                randomize_seq=True,
                                                events_ratio = config["event_ratio"],
                                                delta_t=config["delta_t"],
                                                seq_len=config["seq_len"])
            
        train_loader = DataLoader(train_set,
                                batch_size=config["b"],
                                shuffle=True,
                                collate_fn=yolo_target.collate_fn,
                                num_workers=config["num_workers"],
                                pin_memory=True)
        test_loader = DataLoader(test_set,
                                batch_size=config["b"],
                                shuffle=True,
                                collate_fn=yolo_target.collate_fn,
                                num_workers=config["num_workers"],
                                pin_memory=True)        
    else:
        raise RuntimeError(f'Dataset {config["dataset"]} is not supported.')


    print('Creating YOLO Loss')
    yolo_loss = obd.YOLOLoss(anchors=net.anchors,
                             lambda_coord=config["lambda_coord"],
                             lambda_noobj=config["lambda_noobj"],
                             lambda_obj=config["lambda_obj"],
                             lambda_cls=config["lambda_cls"],
                             lambda_iou=config["lambda_iou"],
                             alpha_iou=config["alpha_iou"],
                             label_smoothing=config["label_smoothing"]).to(device)

    print('Creating Stats Module')
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')

    print('Training/Testing Loop')
    
    for epoch in range(config["epoch"]):
        ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)

        print(f'{epoch=}')
        for i, (inputs, targets, bboxes) in enumerate(train_loader):

            print(f'{i=}') if config["verbose"] else None

            net.train()
            print('inputs') if config["verbose"] else None
            inputs = inputs.to(device)

            print('forward') if config["verbose"] else None
            predictions, counts = net(inputs, sparsity_montior)

            loss, loss_distr = yolo_loss(predictions, targets)
            if sparsity_montior is not None:
                loss += sparsity_montior.loss
                sparsity_montior.clear()

            if torch.isnan(loss):
                print("loss is nan, continuing")
                continue
            optimizer.zero_grad()
            loss.backward()
            net.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config["clip"])
            optimizer.step()
            # scheduler.step()

            # MAP calculations
            T = inputs.shape[-1]
            try:
                predictions = torch.concat([net.yolo(p, a) for (p, a)
                                            in zip(predictions, net.anchors)],
                                           dim=1)
            except RuntimeError:
                print('Runtime error on MAP predictions calculation.'
                      'continuing')
                continue
            predictions = [obd.bbox.utils.nms(predictions[..., t])
                           for t in range(T)]

            for t in range(T):
                ap_stats.update(predictions[t], bboxes[t])

            if not torch.isnan(loss):
                stats.training.loss_sum += loss.item() * inputs.shape[0]
            stats.training.num_samples += inputs.shape[0]
            stats.training.correct_samples = ap_stats[:] * \
                stats.training.num_samples

            processed = i * train_loader.batch_size
            total = len(train_loader.dataset)
            
            header_list = [f'Train: [{processed}/{total} '
                           f'({100.0 * processed / total:.0f}%)]']
            header_list += ['Event Rate: ['
                            + ', '.join([f'{c.item():.2f}'
                                         for c in counts[0]]) + ']']
            header_list += [f'Coord loss: {loss_distr[0].item()}']
            header_list += [f'Obj   loss: {loss_distr[1].item()}']
            header_list += [f'NoObj loss: {loss_distr[2].item()}']
            header_list += [f'Class loss: {loss_distr[3].item()}']
            header_list += [f'IOU   loss: {loss_distr[4].item()}']
            
        if scheduler.__module__ == torch.optim.lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
        else:
            # Using custom defined scheduler
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler(epoch)
                 
        ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)
        for i, (inputs, targets, bboxes) in enumerate(test_loader):
            net.eval()

            with torch.no_grad():
                inputs = inputs.to(device)
                predictions, counts = net(inputs)

                T = inputs.shape[-1]
                predictions = [obd.bbox.utils.nms(predictions[..., t])
                               for t in range(T)]
            
                for t in range(T):
                    ap_stats.update(predictions[t], bboxes[t])

                stats.testing.loss_sum += loss.item() * inputs.shape[0]
                stats.testing.num_samples += inputs.shape[0]
                stats.testing.correct_samples = ap_stats[:] * \
                    stats.testing.num_samples

                processed = i * test_loader.batch_size
                total = len(test_loader.dataset)
                
                header_list = [f'Test: [{processed}/{total} '
                               f'({100.0 * processed / total:.0f}%)]']
                header_list += ['Event Rate: ['
                                + ', '.join([f'{c.item():.2f}'
                                             for c in counts[0]]) + ']']
                header_list += [f'Coord loss: {loss_distr[0].item()}']
                header_list += [f'Obj   loss: {loss_distr[1].item()}']
                header_list += [f'NoObj loss: {loss_distr[2].item()}']
                header_list += [f'Class loss: {loss_distr[3].item()}']
                header_list += [f'IOU   loss: {loss_distr[4].item()}']
                

        os.makedirs("my_model", exist_ok=True)
        torch.save((net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        print("loss ", stats.testing.loss, "accuracy ", stats.testing.accuracy)
        train.report({"loss": stats.testing.loss, "accuracy": stats.testing.accuracy}, checkpoint=checkpoint)
        
        stats.update()
        if stats.testing.best_accuracy is True:
            if hasattr(module, 'export_hdf5'):
                print("best model, saving...")
                torch.save(module.state_dict(), 'my_model/network.pt')
                module.load_state_dict(torch.load("my_model/network.pt"))
                module.export_hdf5("my_model/network.net")         
        
        
    print("Finished Training")
    

def test_best_model(best_result_input):
    
    best_result = best_result_input.config
    
    classes_output = {'BDD100K': 11, 'PropheseeAutomotive': 7}
    print('Using GPUs {}'.format(best_result["gpu"]))
    device = torch.device('cuda:{}'.format(best_result["gpu"][0]))
    
    print('Creating Network')
    if best_result["model"] == 'tiny_yolov3_str':
        Network = obd.models.tiny_yolov3_str.Network
    elif best_result["model"] == 'yolo_kp':
        Network = obd.models.yolo_kp.Network
    elif best_result["model"] == 'tiny_yolov3_str_events':
        Network = obd.models.tiny_yolov3_str_events.Network
    elif best_result["model"] == 'yolo_kp_events':
        Network = obd.models.yolo_kp_events.Network    
    else:
        raise RuntimeError(f'Model type {best_result["model"]=} not supported!')
    
    if len(best_result["gpu"]) == 1:
        net = Network(threshold = best_result["threshold"],
                      tau_grad = best_result["tau_grad"],
                      scale_grad = best_result["scale_grad"],
                      num_classes = classes_output[best_result["dataset"]],
                      clamp_max =best_result["clamp_max"]).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(threshold=best_result["threshold"],
                                            tau_grad=best_result["tau_grad"],
                                            scale_grad=best_result["scale_grad"],
                                            num_classes=classes_output[best_result["dataset"]],
                                            clamp_max=best_result["clamp_max"]).to(device),
                                    device_ids=best_result["gpu"])
        module = net.module
        
    if best_result["model"] == 'tiny_yolov3_str':
       module.init_model((448, 448, 3))
    elif best_result["model"] == 'yolo_kp':
        module.init_model((448, 448, 3))
    elif best_result["model"] == 'tiny_yolov3_str_events':
        module.init_model((448, 448, 2))
    elif best_result["model"] == 'yolo_kp_events':
        module.init_model((448, 448, 2))
    
    checkpoint_path = os.path.join(best_result_input.checkpoint.to_directory(), "checkpoint.pt")

    model_state, _ = torch.load(checkpoint_path)
    net.load_state_dict(model_state)
    
    yolo_target = obd.YOLOtarget(anchors=net.anchors,
                                scales=net.scale,
                                num_classes=net.num_classes,
                                ignore_iou_thres=best_result["tgt_iou_thr"])
    
    if best_result["dataset"] == 'BDD100K':
        test_set = obd.dataset.BDD(root=best_result["path"], dataset='track',
                                   train=False, randomize_seq=True)
        test_loader = DataLoader(test_set,
                                 batch_size=best_result["b"],
                                 shuffle=True,
                                 collate_fn=yolo_target.collate_fn,
                                 num_workers=best_result["num_workers"],
                                 pin_memory=True)        
    elif best_result["dataset"] == 'PropheseeAutomotive':
        if best_result["subset"]:   
            test_set = PropheseeAutomotiveSmallTrain(root=best_result["path"], train=False, 
                                                augment_prob=best_result["aug_prob"], 
                                                randomize_seq=True,
                                                events_ratio = best_result["event_ratio"],
                                                delta_t=best_result["delta_t"],
                                                seq_len=best_result["seq_len"],
                                                reduce_classes=True)
            print('Using PropheseeAutomotiveSmall Dataset')
        else:     
            test_set = obd.dataset.PropheseeAutomotive(root=best_result["path"], train=False, 
                                                augment_prob=best_result["aug_prob"], 
                                                randomize_seq=True,
                                                events_ratio = best_result["event_ratio"],
                                                delta_t=best_result["delta_t"],
                                                seq_len=best_result["seq_len"])
        test_loader = DataLoader(test_set,
                                batch_size=best_result["b"],
                                shuffle=True,
                                collate_fn=yolo_target.collate_fn,
                                num_workers=best_result["num_workers"],
                                pin_memory=True)        
    else:
        raise RuntimeError(f'Dataset {best_result["dataset"]} is not supported.')
    

    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')
    ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)

    for i, (inputs, targets, bboxes) in enumerate(test_loader):
        net.eval()

        with torch.no_grad():
            inputs = inputs.to(device)
            predictions, counts = net(inputs)

            T = inputs.shape[-1]
            
            predictions = [obd.bbox.utils.nms(predictions[..., t]) for t in range(T)]

            for t in range(T):
                ap_stats.update(predictions[t], bboxes[t])

            stats.testing.num_samples += inputs.shape[0]
            stats.testing.correct_samples = ap_stats[:] * stats.testing.num_samples

            processed = i * test_loader.batch_size
            total = len(test_loader.dataset)
    print("Best trial test set accuracy: {}".format(processed / total))


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1, cpus_per_trial=4):
    config = {
        'gpu': [0],
        'b': 1,
        'verbose': False,
        # Model
        'model': 'yolo_kp_events',
        # Sparsity
        'sparsity': False,
        'sp_lam': 0.01,
        'sp_rate': 0.01,
        # Optimizer
        'lr': tune.loguniform(1e-4, 1e-2),
        'wd': 1e-5,
        #'lrf': tune.loguniform(1e-4, 1e-1),
        # Network/SDNN
        'threshold': 0.1,
        'tau_grad':  0.1,
        'scale_grad': 1.0,
        'clip': 10,
        # Target generation
        'tgt_iou_thr': 0.5,
        # YOLO loss
        'lambda_coord': tune.loguniform(0.01, 30.0),
        'lambda_noobj': tune.loguniform(0.01, 30.0),
        'lambda_obj': tune.loguniform(0.01, 30.0),
        'lambda_cls': tune.loguniform(0.01, 30.0),
        'lambda_iou':tune.loguniform(0.01, 30.0),
        'alpha_iou': tune.loguniform(0.01, 1.0),
        'label_smoothing': tune.uniform(0.01, 1.0),
        'track_iter': 1000,
        # Training
        'epoch': 2, #1000,
        'warmup': 0,
        # dataset
        'dataset': 'PropheseeAutomotive',
        'subset': True,
        'seq_len': 32,
        'delta_t': 10,
        'event_ratio': 0.04,
        'path': '/home/lecampos/data/prophesee',
        'output_dir': '.',
        'num_workers': 4,
        'aug_prob': 0.0,
        'clamp_max': 5.0,
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        metric="accuracy",
        mode = "max",
        grace_period=2,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_events),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            #metric="accuracy",
            #mode="max",
            scheduler=scheduler,
            num_samples=num_samples
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("accuracy", "max", scope="last-5-avg")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

#main(num_samples=30, max_num_epochs=50, gpus_per_trial=1, cpus_per_trial=80)

main(num_samples=5, max_num_epochs=5, gpus_per_trial=1)