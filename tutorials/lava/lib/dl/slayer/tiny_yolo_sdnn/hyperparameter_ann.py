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


def train_ann(config):
    classes_output = {'BDD100K': 11}
    
    print('Using GPUs {}'.format(config["gpu"]))
    device = torch.device('cuda:{}'.format(config["gpu"][0]))
    
    print('Creating Network')
    if config["model"] == 'yolov3_ann':
        Network = obd.models.yolov3_ann.Network
        yolo_anchors =  [ 
                        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
                        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
                        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
                    ]
    elif config["model"] == 'tiny_yolov3_ann':
        Network = obd.models.tiny_yolov3_ann.Network
        yolo_anchors = [[(0.28, 0.22), 
                    (0.38, 0.48), 
                    (0.9, 0.78)], 
                [(0.07, 0.15), 
                    (0.15, 0.11), 
                    (0.14, 0.29)]] if config["model_type"] != 'single-head' else  \
                [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]]
    else:
        raise RuntimeError(f'Model type {config["model"]=} not supported!')
        
    if len(config["gpu"]) == 1:
        net = Network(num_classes=classes_output[config['dataset']], 
                      yolo_type=config['model_type'],
                      anchors=yolo_anchors).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(num_classes=classes_output[config['dataset']], 
                                            yolo_type=config['model_type'],
                                            anchors=yolo_anchors).to(device),
                                    device_ids=config["gpu"])
        module = net.module
    
    module.init_model((448, 448, 3))

    # Define optimizer module.
    print('Creating Optimizer')
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=config["lr"],
                                 weight_decay=config["wd"])

    # Define learning rate scheduler
    def lf(x):
        return (min(x / config["warmup"], 1)
                * ((1 + np.cos(x * np.pi / config["epoch"])) / 2)
                * (1 - config["lrf"])
                + config["lrf"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

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
        for i, (inputs_t, targets_t, bboxes_t) in enumerate(train_loader):
            for idx in range(inputs_t.shape[4]):
                print(f'{i=}') if config["verbose"] else None
                net.train()
                print('inputs') if config["verbose"] else None
                inputs = inputs_t[...,idx].to(device)
                print('forward') if config["verbose"] else None
                predictions, _ = net(inputs)
                
                targets = [item[..., idx, None] for item in targets_t]
                predictions = [item[..., None] for item in predictions]
                loss, loss_distr = yolo_loss(predictions, targets)

                if torch.isnan(loss):
                    print("loss is nan, continuing")
                    continue
                optimizer.zero_grad()
                loss.backward()
                net.validate_gradients()
                optimizer.step()
                # MAP calculations
                try:
                    predictions = torch.concat([net.yolo(p, a) for (p, a)
                                                in zip(predictions, net.anchors)],
                                            dim=1)
                except RuntimeError:
                    print('Runtime error on MAP predictions calculation.'
                        'continuing')
                    continue
                predictions = obd.bbox.utils.nms(predictions[..., 0])
                ap_stats.update(predictions, bboxes_t[idx])

                if not torch.isnan(loss):
                    stats.training.loss_sum += loss.item() * inputs.shape[0]
                stats.training.num_samples += inputs.shape[0]
                stats.training.correct_samples = ap_stats[:] * \
                    stats.training.num_samples

        scheduler.step()
        ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)
        
        for i, (inputs_t, targets_t, bboxes_t) in enumerate(test_loader):
            net.eval()
            predictions_t = []
            for idx in range(inputs_t.shape[4]):
                with torch.no_grad():
                    inputs = inputs_t[...,idx].to(device)
                    inputs = inputs.to(device)
                    
                    predictions, _ = net(inputs)
                    
                    predictions = obd.bbox.utils.nms(predictions[..., 0])
                    predictions_t.append(predictions)
                    ap_stats.update(predictions, bboxes_t[idx])

                    stats.testing.loss_sum += loss.item() * inputs.shape[0]
                    stats.testing.num_samples += inputs.shape[0]
                    stats.testing.correct_samples = ap_stats[:] * \
                        stats.testing.num_samples
                    
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
    classes_output = {'BDD100K': 11}
    
    print('Using GPUs {}'.format(best_result["gpu"]))
    device = torch.device('cuda:{}'.format(best_result["gpu"][0]))
    
    print('Creating Network')
    if best_result["model"] == 'yolov3_ann':
        Network = obd.models.yolov3_ann.Network
        yolo_anchors =  [ 
                        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
                        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
                        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
                    ]
    elif best_result["model"] == 'tiny_yolov3_ann':
        Network = obd.models.tiny_yolov3_ann.Network
        yolo_anchors = [[(0.28, 0.22), 
                    (0.38, 0.48), 
                    (0.9, 0.78)], 
                [(0.07, 0.15), 
                    (0.15, 0.11), 
                    (0.14, 0.29)]] if best_result["model_type"] != 'single-head' else  \
                [[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]]
    else:
        raise RuntimeError(f'Model type {best_result["model"]=} not supported!')
    
    if len(best_result["gpu"]) == 1:
        net = Network(num_classes=classes_output[best_result['dataset']], 
                      yolo_type=best_result['model_type'],
                      anchors=yolo_anchors).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(num_classes=classes_output[best_result['dataset']], 
                                            yolo_type=best_result['model_type'],
                                            anchors=yolo_anchors).to(device),
                                    device_ids=best_result["gpu"])
        module = net.module

    module.init_model((448, 448, 3))
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
    else:
        raise RuntimeError(f'Dataset {best_result["dataset"]} is not supported.')
    
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')
    ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)
    for i, (inputs_t, targets_t, bboxes_t) in enumerate(test_loader):
        net.eval()
        predictions_t = []
        for idx in range(inputs_t.shape[4]):
            with torch.no_grad():
                inputs = inputs_t[...,idx].to(device)
                inputs = inputs.to(device)
                
                predictions, _ = net(inputs)
                
                predictions = obd.bbox.utils.nms(predictions[..., 0])
                predictions_t.append(predictions)
                ap_stats.update(predictions, bboxes_t[idx])

                stats.testing.num_samples += inputs.shape[0]
                stats.testing.correct_samples = ap_stats[:] * \
                    stats.testing.num_samples
    print("Best trial test set accuracy: {}".format(stats.testing.accuracy))


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1, cpus_per_trial=4,
         model='tiny_yolov3_ann', model_type='single-head'):
    config = {
        'gpu': [0],
        'b': 16,
        'verbose': False,
        # Model
        'model': model, # yolov3_ann tiny_yolov3_ann
        'model_type': model_type, # complete, str, single-head
        # Optimizer
        'lr': tune.loguniform(1e-4, 1e-2),
        'wd': 1e-5,
        'lrf': tune.loguniform(1e-2, 1e-3),
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
        'epoch': 500,
        'warmup': 10,
        # dataset
        'dataset': 'BDD100K',
        'path': '/home/lecampos/data/bdd100k',
        'output_dir': '.',
        'num_workers': 8,
        'aug_prob': 0.2
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        metric="accuracy",
        mode = "max",
        grace_period=20,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_ann),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=3,
        ),
        param_space=config,
        run_config=ray.train.RunConfig(name="train_ann_" + model + '_' + model_type)
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


if __name__ == '__main__':
    samples = 40
    main(num_samples=samples, max_num_epochs=500, gpus_per_trial=1, cpus_per_trial=9,
        model='tiny_yolov3_ann', model_type='complete')
    main(num_samples=samples, max_num_epochs=500, gpus_per_trial=1, cpus_per_trial=9,
        model='tiny_yolov3_ann', model_type='str')
    main(num_samples=samples, max_num_epochs=500, gpus_per_trial=1, cpus_per_trial=9,
        model='tiny_yolov3_ann', model_type='single-head')
    main(num_samples=samples, max_num_epochs=500, gpus_per_trial=1, cpus_per_trial=9,
        model='yolov3_ann')
