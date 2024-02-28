import torch
import torch_tensorrt
import tensorrt
import os
from torch.utils.data import DataLoader
import time
import torch.backends.cudnn as cudnn
import numpy as np
import threading

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib
from tqdm import tqdm
import subprocess
import sys
sys.path.append("/home/lecampos/elrond91/lava-dl/src")
#sys.path.append("/home/lecampos/lava-nc/lava-dl/src/")

from lava.lib.dl.slayer import obd
from lava.lib.dl import slayer

torch_tensorrt.logging.debug()

print("tensorrt ", tensorrt.__version__)
print("torch_tensorrt ",torch_tensorrt.__version__)
print("pytorch_quantization ", pytorch_quantization.__version__)


def yolo(x: torch.tensor, anchors: torch.tensor, quantize: bool = False, clamp_max: float = 5.0) -> torch.tensor:
    N, _, _, _, P, T = x.shape
    return obd.yolo_base._yolo(x, anchors, clamp_max, quantize).reshape([N, -1, P, T])

def measuse_power(debug=False):
    try:
        result = subprocess.Popen(['sudo', '-S'] + './tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/jtop_stats.py'.split(),
                                stdout=subprocess.PIPE)
        
        out, _ = result.communicate()
        info = out.decode("utf-8").replace("\'", "\"")
        if debug:
            print(info)
            print(info.split("RAM")[1].split("}")[0].split("tot")[1].split(",")[0].split(":")[1])
        VDD_CPU_GPU_CV = int(info.split("VDD_CPU_GPU_CV")[1].split("}")[0].split("{")[1].split("power")[1].split(",")[0].split(":")[1])
        VDD_SOC = int(info.split("VDD_SOC")[1].split("}")[0].split("{")[1].split("power")[1].split(",")[0].split(":")[1])
        tot = int(info.split("tot")[1].split("}")[0].split("{")[1].split("power")[1].split(",")[0].split(":")[1])
        GPU_PERCENT = float(info.split("load")[1].split("}")[0].split(":")[1])
        TOTAL_RAM = int(info.split("RAM")[1].split("}")[0].split("tot")[1].split(",")[0].split(":")[1])
        USED_RAM = int(info.split("RAM")[1].split("}")[0].split("used")[1].split(",")[0].split(":")[1])
        FREE_RAM = int(info.split("RAM")[1].split("}")[0].split("free")[1].split(",")[0].split(":")[1])
        BUFFERS_RAM = int(info.split("RAM")[1].split("}")[0].split("buffers")[1].split(",")[0].split(":")[1])
        CACHED_RAM = int(info.split("RAM")[1].split("}")[0].split("cached")[1].split(",")[0].split(":")[1])
        SHARED_RAM = int(info.split("RAM")[1].split("}")[0].split("shared")[1].split(",")[0].split(":")[1])
        SWAP_RAM = int(info.split("SWAP")[1].split("}")[0].split("tot")[1].split(",")[0].split(":")[1])
        # add % GPU
        return time.time(), VDD_CPU_GPU_CV, VDD_SOC, tot, GPU_PERCENT, TOTAL_RAM, USED_RAM, FREE_RAM, BUFFERS_RAM, CACHED_RAM, SHARED_RAM, SWAP_RAM
    except:
        return None

class JTOPStats:
    def __init__(self):
        self._stats = []
        self._stats_file = []
        self._stop = False
        self._thread = threading.Thread(target=self.adquire)
        self._thread.start()

    def adquire(self):
        while not self._stop:
            results = measuse_power()
            if results is not None:
                self._stats_file.append(results)
                self._stats.append(self._stats_file[-1][1:])
            time.sleep(1)

    def stop(self):
        self._stop = True
        self._thread.join()
    
    def get_stats(self, file = None):
        if file is not None:
            head = ["time, VDD_CPU_GPU_CV, VDD_SOC, tot, GPU_PERCENT, TOTAL_RAM, USED_RAM, FREE_RAM, BUFFERS_RAM, CACHED_RAM, SHARED_RAM, SWAP_RAM"]
            with open(file, 'w') as f:
                f.write(f"{head}\n")
                for line in self._stats_file:
                    f.write(f"{line}\n")
        return np.mean(np.asarray(self._stats), axis=0)

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (input_img, _, _) in tqdm(enumerate(data_loader), total=num_batches):
        for idx in range(input_img.shape[4]):
            image = input_img[...,idx]
            model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: classification model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """
    model.eval()
    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

def benchmark(model, dataloader, anchors, clamp_max, batch_size, quantize = False, save_exp = None):
    print("Start benchmark ...")
    all_time = time.time()
    anchors = torch.tensor(anchors).cuda()
    
    time_computing = [0, 0, 0]
    time_cumulator = 0
    
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')
    ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)
    jstats = JTOPStats()
    for idx in range(30):
        print('iteration... ', idx)
        for i, (inputs_t, targets_t, bboxes_t) in enumerate(dataloader):
            if inputs_t.shape[0] != batch_size:
                continue
            model.eval()
            predictions_t = []
            for idx in range(inputs_t.shape[4]):
                with torch.no_grad():
                    inputs = inputs_t[...,idx]
                    
                    start_time = time.time()
                    inputs = inputs.cuda()
                    time_computing[0] += time.time() - start_time
                    
                    if quantize:
                        start_time = time.time()
                        predictions = model(inputs)
                        time_computing[1] += time.time() - start_time
                    else:
                        start_time = time.time()
                        predictions, _ = model(inputs)
                        time_computing[1] += time.time() - start_time
                    
                    start_time = time.time()
                    to_cpu = [item.cpu().data for item in predictions]
                    time_computing[2] += time.time() - start_time
                    
                    time_cumulator += 1
                        
                    predictions = [item[..., None] for item in predictions]
                    predictions = torch.concat([yolo(p, a, quantize=quantize, clamp_max=clamp_max) for (p, a) in zip(predictions,  anchors)], dim=1)
                    predictions = obd.bbox.utils.nms(predictions[..., 0])
                    predictions_t.append(predictions)
                    ap_stats.update(predictions, bboxes_t[idx])
                    stats.testing.num_samples += inputs.shape[0]
                    stats.testing.correct_samples = ap_stats[:] * \
                        stats.testing.num_samples
    jstats.stop()     
    if save_exp is None:
        result_stats = jstats.get_stats()
    else:
        result_stats = jstats.get_stats(save_exp[0] + save_exp[1] + '_gpu_stats_time.txt')
        with open(save_exp[0] + save_exp[1] + '_stats.txt', 'w') as f:
            f.write("VDD_CPU_GPU_CV: {:.5f}milliwatt; VDD_SOC: {:.5f}milliwatt; tot: {:.5f}milliwatt perc: {:.5f}\n".format(result_stats[0], result_stats[1], result_stats[2], result_stats[3]))
            f.write('Average data input time : %.5f ms \n'%( (time_computing[0] / time_cumulator) * 1000))
            f.write('Average inference time: %.5f ms \n'%( (time_computing[1] / time_cumulator) * 1000))
            f.write('Average data output time: %.5f ms \n'%( (time_computing[2] / time_cumulator) * 1000))
            f.write('Benchmark time: %.5f s \n'%( (time.time() - all_time)))
            f.write("Accuracy: {:.5f}".format(stats.testing.accuracy))
            
    print("VDD_CPU_GPU_CV: {:.5f}milliwatt; VDD_SOC: {:.5f}milliwatt; tot: {:.5f}milliwatt perc: {:.5f}".format(result_stats[0], result_stats[1], result_stats[2], result_stats[3]))          
    print('Average data input time : %.5f ms \n'%( (time_computing[0] / time_cumulator) * 1000))
    print('Average inference time: %.5f ms \n'%( (time_computing[1] / time_cumulator) * 1000))
    print('Average data output time: %.5f ms \n'%( (time_computing[2] / time_cumulator) * 1000))
    print('Benchmark time: %.5f s \n'%( (time.time() - all_time)))
    return stats.testing.loss, stats.testing.accuracy


def benchmark_model(config, folder_model, batch_size = 1):
    
    print('Starting exp: ' + str(batch_size))
    print('Using GPUs {}'.format(config["gpu"]))
    device = torch.device('cuda:{}'.format(config["gpu"]))
    
    os.makedirs(folder_model + '/profiling', exist_ok=True)
    
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

    net = Network(num_classes=11, 
                    yolo_type=config['model_type'],
                    anchors=yolo_anchors).to(device)
    module = net
    module.init_model((448, 448, 3))
    
    print('Loading Dataset')
    yolo_target = obd.YOLOtarget(anchors=net.anchors,
                                scales=net.scale,
                                num_classes=net.num_classes,
                                ignore_iou_thres=config["tgt_iou_thr"])
    
    train_set = obd.dataset.BDD(root='/home/lecampos/data/bdd100k', dataset='track',
                                train=True, augment_prob=0.2,
                                seq_len=1,
                                randomize_seq=True)
    test_set = obd.dataset.BDD(root='/home/lecampos/data/bdd100k', dataset='track',
                                train=False, 
                                seq_len=1,
                                randomize_seq=True)
    
    train_loader = DataLoader(train_set,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=yolo_target.collate_fn,
                                num_workers=1,
                                pin_memory=True)
    test_loader = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=yolo_target.collate_fn,
                                num_workers=1,
                                pin_memory=True)
    print('train_loader: ', len(train_loader))
    print('test_loader: ', len(test_loader))
    print('Loading weights')
    module.load_state_dict(torch.load(folder_model + '/network.pt', 
                                      map_location= 'cuda:{}'.format(config["gpu"])))
    clamp_max = module.clamp_max
    save_experiment = [folder_model, '/profiling/' + config["model"] + "_" + config['model_type'] + "_" + str(batch_size) + "_float"]
    cudnn.benchmark = True
    loss, accuracy = benchmark(net, test_loader, yolo_anchors, clamp_max, batch_size=batch_size, save_exp=save_experiment)
    print("Float " + config["model"] + "_" + config['model_type'] +" loss: {:.5f} accuracy: {:.5f}".format(loss, accuracy))

    del module
    del net

    qat_model = Network(num_classes=11, 
                      yolo_type=config['model_type'],
                      anchors=yolo_anchors, quantize=True).to(device)
    qat_model = qat_model.cuda()
    qat_model.init_model((448, 448, 3))
    
    qat_model.load_state_dict(torch.load(folder_model + '/network.pt', 
                                      map_location= 'cuda:{}'.format(config["gpu"])))
    
    #calibrate if model not found
    if not os.path.exists(folder_model + "/" + config["model"] + "_" + config['model_type']+ "_qat.jit.pt"):
        #Calibrate the model using max calibration technique.
        with torch.no_grad():
            calibrate_model(
                model=qat_model,
                model_name=config["model"] + '_' + config['model_type'],
                data_loader=train_loader,
                num_calib_batch=int(len(train_loader)),
                calibrator="max",
                hist_percentile=[99.9, 99.99, 99.999, 99.9999],
                out_dir=folder_model + '/')
            
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        with torch.no_grad():
            data_in = iter(test_loader)
            images_raw, _, _ = next(data_in)
            images = images_raw[...,0]
            jit_model = torch.jit.trace(qat_model, images.to("cuda")).eval()
            torch.jit.save(jit_model, folder_model + "/" + config["model"] + "_" + config['model_type']+ "_qat.jit.pt")
    else:
       jit_model =  torch.jit.load(folder_model + "/" + config["model"] + "_" + config['model_type']+ "_qat.jit.pt").eval()
    
    qat_model = torch.jit.load(folder_model + "/" + config["model"] + "_" + config['model_type']+ "_qat.jit.pt").eval()
    
    del train_loader
    del train_set
    
    save_experiment = [folder_model, '/profiling/' +config["model"] + "_" + config['model_type'] + "_" + str(batch_size) + "_jit"]
    loss, accuracy = benchmark(jit_model, test_loader, yolo_anchors, clamp_max, batch_size=batch_size, quantize = True, save_exp=save_experiment)
    print("jit_model " + config["model"] + "_" + config['model_type'] +" loss: {:.5f} accuracy: {:.5f}".format(loss, accuracy))

    del jit_model
    
    data_in = iter(test_loader)
    images_raw, _, _ = next(data_in)
    images = images_raw[...,0]
    B, C, W, H = images.shape

    compile_spec = {"inputs": [torch_tensorrt.Input([B, C, W, H])],
                    "enabled_precisions": torch.int8,
                    "truncate_long_and_double": True,
                    }
    trt_mod = torch_tensorrt.compile(qat_model, **compile_spec)
    
    del qat_model
    save_experiment = [folder_model, '/profiling/' +config["model"] + "_" + config['model_type'] + "_" + str(batch_size) + "_trt"]
    loss, accuracy = benchmark(trt_mod, test_loader, yolo_anchors, clamp_max, batch_size=batch_size, quantize= True, save_exp=save_experiment)
    print("trt_mod " + config["model"] + "_" + config['model_type'] +" loss: {:.5f} accuracy: {:.5f}".format(loss, accuracy))
    
    

if __name__ == '__main__':
    config = {
        'gpu': 0,
        # Model
        'model': 'tiny_yolov3_ann', # yolov3_ann tiny_yolov3_ann
        'model_type': 'single-head', # complete, str, single-head
        # Target generation
        'tgt_iou_thr': 0.5,
    }
    base_path = '/home/lecampos/models/'
    #benchmark_model(config, base_path + 'Trained_tiny_yolov3_ann_single-head', batch_size=1)
    benchmark_model(config, base_path + 'Trained_tiny_yolov3_ann_single-head', batch_size=2)
    benchmark_model(config, base_path + 'Trained_tiny_yolov3_ann_single-head', batch_size=4)
    benchmark_model(config, base_path + 'Trained_tiny_yolov3_ann_single-head', batch_size=6)
    benchmark_model(config, base_path + 'Trained_tiny_yolov3_ann_single-head', batch_size=8)
    benchmark_model(config, base_path + 'Trained_tiny_yolov3_ann_single-head', batch_size=16)
    benchmark_model(config, base_path + 'Trained_tiny_yolov3_ann_single-head', batch_size=32)
    
    # print(measuse_power(debug=True))
    
    # 1) single-head [batch = 1; batch = [2, 4, 8, 16, 32, 64]] -> tensorRT + quant
    # 2) tiny_yolov3 [batch = 1; batch = [2, 4, 8, 16, 32, 64]] -> tensorRT + quant
    # https://developer.nvidia.com/embedded/jetson-benchmarks
    # 3) tiny_yolov3_str [batch = 1; batch = [2, 4, 8, 16, 32, 64]] -> tensorRT + quant
    # ------
    # 4) yolov3 [batch = 1; batch = [2, 4, 8, 16, 32, 64]] -> tensorRT + quant
   

    
    