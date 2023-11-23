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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=[0], help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',   type=int, default=12,  help='batch size for dataloader')
    parser.add_argument('-verbose', default=False, action='store_true', help='lots of debug printouts')
    # Model
    parser.add_argument('-model', type=str, default='yolov3_ann', help='network model')
    # Optimizer
    parser.add_argument('-lr',  type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-wd',  type=float, default=1e-5,   help='optimizer weight decay')
    parser.add_argument('-lrf', type=float, default=0.01,   help='learning rate reduction factor for lr scheduler')
    # Pretrained model
    parser.add_argument('-load', type=str, default='', help='pretrained model')
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
    parser.add_argument('-dataset',     type=str,   default='BDD100K', help='dataset to use [BDD100K, PropheseeAutomotive]')
    parser.add_argument('-path',        type=str,   default='/home/lecampos/data/bdd100k', help='dataset path')
    parser.add_argument('-output_dir',  type=str,   default='.', help='directory in which to put log folders')
    parser.add_argument('-num_workers', type=int,   default=6, help='number of dataloader workers')
    parser.add_argument('-aug_prob',    type=float, default=0.2, help='training augmentation probability')

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
    if args.model == 'yolov3_ann':
        Network = obd.models.yolov3_ann.Network
    else:
        raise RuntimeError(f'Model type {args.model=} not supported!')
    
    if len(args.gpu) == 1:
        net = Network(num_classes=classes_output[args.dataset]).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(num_classes=classes_output[args.dataset]).to(device), device_ids=args.gpu)
        module = net.module

    print('Loading Network')
    if args.load != '':
        saved_model = args.load
        print(f'Initializing model from {saved_model}')
        module.load_model(saved_model)
        
    module.init_model((448, 448, 3))

    # Define optimizer module.
    print('Creating Optimizer')
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)

    # Define learning rate scheduler
    def lf(x):
        return (min(x / args.warmup, 1)
                * ((1 + np.cos(x * np.pi / args.epoch)) / 2)
                * (1 - args.lrf)
                + args.lrf)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    yolo_target = obd.YOLOtarget(anchors=net.anchors,
                                 scales=net.scale,
                                 num_classes=net.num_classes,
                                 ignore_iou_thres=args.tgt_iou_thr)

    print('Creating Dataset')

    if args.dataset == 'BDD100K':
        train_set = obd.dataset.BDD(root=args.path, dataset='track',
                                    train=True, augment_prob=args.aug_prob,
                                    randomize_seq=True)
        test_set = obd.dataset.BDD(root=args.path, dataset='track',
                                   train=False, randomize_seq=True)
        train_loader = DataLoader(train_set,
                                  batch_size=args.b,
                                  shuffle=True,
                                  collate_fn=yolo_target.collate_fn,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
        test_loader = DataLoader(test_set,
                                 batch_size=args.b,
                                 shuffle=True,
                                 collate_fn=yolo_target.collate_fn,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
    else:
        raise RuntimeError(f'Dataset {args.dataset} is not supported.')

    box_color_map = [(np.random.randint(256),
                      np.random.randint(256),
                      np.random.randint(256))
                     for _ in range(classes_output[args.dataset])]

    print('Creating YOLO Loss')
    yolo_loss = obd.YOLOLoss(anchors=net.anchors,
                             lambda_coord=args.lambda_coord,
                             lambda_noobj=args.lambda_noobj,
                             lambda_obj=args.lambda_obj,
                             lambda_cls=args.lambda_cls,
                             lambda_iou=args.lambda_iou,
                             alpha_iou=args.alpha_iou,
                             label_smoothing=args.label_smoothing).to(device)

    print('Creating Stats Module')
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')

    loss_tracker = dict(coord=[], obj=[], noobj=[], cls=[], iou=[])
    loss_order = ['coord', 'obj', 'noobj', 'cls', 'iou']

    print('Training/Testing Loop')
    
    for epoch in range(args.epoch):
        t_st = datetime.now()
        ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)

        print(f'{epoch=}')
        for i, (inputs_t, targets_t, bboxes_t) in enumerate(train_loader):
            for idx in range(inputs_t.shape[4]):

                print(f'{i=}') if args.verbose else None

                net.train()
                print('inputs') if args.verbose else None
                inputs = inputs_t[...,idx].to(device)

                print('forward') if args.verbose else None
                predictions, _ = net(inputs)
                
                # add extra dimenstion! predictions, targets
                targets = [item[..., idx, None] for item in targets_t]
                predictions = [item[..., None] for item in predictions]
                loss, loss_distr = yolo_loss(predictions, targets)

                if torch.isnan(loss):
                    print("loss is nan, continuing")
                    continue
                optimizer.zero_grad()
                loss.backward()
                net.validate_gradients()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                scheduler.step()

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

                processed = i * train_loader.batch_size
                total = len(train_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
                header_list = [f'Train: [{processed}/{total} '
                            f'({100.0 * processed / total:.0f}%)]']
                header_list += [f'Coord loss: {loss_distr[0].item()}']
                header_list += [f'Obj   loss: {loss_distr[1].item()}']
                header_list += [f'NoObj loss: {loss_distr[2].item()}']
                header_list += [f'Class loss: {loss_distr[3].item()}']
                header_list += [f'IOU   loss: {loss_distr[4].item()}']

                if i % args.track_iter == 0:
                    plt.figure()
                    for loss_idx, loss_key in enumerate(loss_order):
                        loss_tracker[loss_key].append(loss_distr[loss_idx].item())
                        plt.semilogy(loss_tracker[loss_key], label=loss_key)
                        writer.add_scalar(f'Loss Tracker/{loss_key}',
                                            loss_distr[loss_idx].item(),
                                            len(loss_tracker[loss_key]) - 1)
                    plt.xlabel(f'iters (x {args.track_iter})')
                    plt.legend()
                    plt.savefig(f'{trained_folder}/yolo_loss_tracker.png')
                    plt.close()
                stats.print(epoch, i, samples_sec, header=header_list)

        t_st = datetime.now()
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

                    processed = i * test_loader.batch_size
                    total = len(test_loader.dataset)
                    time_elapsed = (datetime.now() - t_st).total_seconds()
                    samples_sec = time_elapsed / (i + 1) / test_loader.batch_size
                    header_list = [f'Test: [{processed}/{total} '
                                f'({100.0 * processed / total:.0f}%)]']
                    header_list += [f'Coord loss: {loss_distr[0].item()}']
                    header_list += [f'Obj   loss: {loss_distr[1].item()}']
                    header_list += [f'NoObj loss: {loss_distr[2].item()}']
                    header_list += [f'Class loss: {loss_distr[3].item()}']
                    header_list += [f'IOU   loss: {loss_distr[4].item()}']
                    stats.print(epoch, i, samples_sec, header=header_list)
    
        writer.add_scalar('Loss/train', stats.training.loss, epoch)
        writer.add_scalar('mAP@50/train', stats.training.accuracy, epoch)
        writer.add_scalar('mAP@50/test', stats.testing.accuracy, epoch)

        stats.update()
        stats.plot(path=trained_folder + '/')
        
        b = -1        
        image = Image.fromarray(np.uint8(
            inputs[b, :, :, :].cpu().data.numpy().transpose([1, 2, 0]) * 255
        ))
            
        annotation = obd.bbox.utils.annotation_from_tensor(
            predictions[0][b],
            {'height': image.height, 'width': image.width},
            test_set.classes,
            confidence_th=0
        )
        marked_img = obd.bbox.utils.mark_bounding_boxes(
            image, annotation['annotation']['object'],
            box_color_map=box_color_map, thickness=5
        )
        
        image = Image.fromarray(np.uint8(
            inputs[b, :, :, :].cpu().data.numpy().transpose([1, 2, 0]) * 255
        ))

        annotation = obd.bbox.utils.annotation_from_tensor(
            bboxes_t[0][b],
            {'height': image.height, 'width': image.width},
            test_set.classes,
            confidence_th=0
        )
        marked_gt = obd.bbox.utils.mark_bounding_boxes(
            image, annotation['annotation']['object'],
            box_color_map=box_color_map, thickness=5
        )

        marked_images = Image.new('RGB', (marked_img.width + marked_gt.width,
                                          marked_img.height))
        marked_images.paste(marked_img, (0, 0))
        marked_images.paste(marked_gt, (marked_img.width, 0))

        writer.add_image('Prediction',
                            transforms.PILToTensor()(marked_images),
                            epoch)
        if stats.testing.best_accuracy is True:
            torch.save(module.state_dict(), trained_folder + '/network.pt')
            if inputs.shape[-1] == 1:
                marked_images.save(
                    f'{trained_folder}/prediction_{epoch}_{b}.jpg')
            else:
                filename = f'{trained_folder}/prediction_{epoch}_{b}'
                obd.bbox.utils.create_video(inputs_t, bboxes_t, predictions_t,
                                    filename, test_set.classes,
                                    box_color_map=box_color_map)
                    
        stats.save(trained_folder + '/')

    params_dict = {}
    for key, val in args._get_kwargs():
        params_dict[key] = str(val)
    writer.add_hparams(params_dict, {'mAP@50': stats.testing.max_accuracy})
    writer.flush()
    writer.close()
