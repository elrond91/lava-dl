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


def measuse_power():
    result = subprocess.Popen(['sudo', '-S'] + './tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/jtop_stats.py'.split(),
                            stdout=subprocess.PIPE)
    
    out, _ = result.communicate()
    info = out.decode("utf-8").replace("\'", "\"")
    VDD_CPU_GPU_CV = int(info.split("VDD_CPU_GPU_CV")[1].split("}")[0].split("{")[1].split("power")[1].split(",")[0].split(":")[1])
    VDD_SOC = int(info.split("VDD_SOC")[1].split("}")[0].split("{")[1].split("power")[1].split(",")[0].split(":")[1])
    tot = int(info.split("tot")[1].split("}")[0].split("{")[1].split("power")[1].split(",")[0].split(":")[1])
    # add % GPU
    return VDD_CPU_GPU_CV, VDD_SOC, tot

class JTOPStats:
    def __init__(self):
        self._stats = []
        self._stop = False
        self._thread = threading.Thread(target=self.adquire)
        self._thread.start()

    def adquire(self):
        while not self._stop:
            self._stats.append(measuse_power())
            time.sleep(10)

    def stop(self):
        self._stop = True
        self._thread.join()
    
    def get_stats(self):
        print(np.asarray(self._stats))
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

def test(model, dataloader):
    print("Start testing ...")
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')
    ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)
    jstats = JTOPStats()
    for idx in range(5):
        print(idx)
        for i, (inputs_t, targets_t, bboxes_t) in enumerate(dataloader):
            model.eval()
            predictions_t = []
            
            for idx in range(inputs_t.shape[4]):
                with torch.no_grad():
                    inputs = inputs_t[...,idx]
                    inputs = inputs.cuda()
                    
                    predictions = model(inputs)
                    
                    predictions = obd.bbox.utils.nms(predictions[..., 0])
                    predictions_t.append(predictions)
                    ap_stats.update(predictions, bboxes_t[idx])

                    stats.testing.num_samples += inputs.shape[0]
                    stats.testing.correct_samples = ap_stats[:] * \
                        stats.testing.num_samples
    jstats.stop()
    result_stats = jstats.get_stats()
    print("VDD_CPU_GPU_CV: {:.5f} VDD_SOC: {:.5f} tot: {:.5f} ".format(result_stats[0], result_stats[1], result_stats[2]))          
    return stats.testing.loss, stats.testing.accuracy

# Helper function to benchmark the model
def benchmark(model, dataloader, batch_size, dtype='fp32'):
    print("Start timing ...")
    time_cum = 0
    times_computed = 0
    with torch.no_grad():
        jstats = JTOPStats()
        for idx in range(5):
            print(idx)
            for data, _, _ in dataloader:
                for idx in range(data.shape[4]):
                    inputs = data[...,idx]
                    if inputs.shape[0] != batch_size:
                        continue
                    start_time = time.time()
                    inputs = inputs.cuda()
                    output = model(inputs)
                    output = output.cpu().data
                    time_cum += time.time() - start_time
                    times_computed += 1
        jstats.stop()
        result_stats = jstats.get_stats()
        print("Output shape:", output.shape)
        print('Average data time: %.5f ms'%( (time_cum / times_computed) * 1000))
        print("VDD_CPU_GPU_CV: {:.5f} VDD_SOC: {:.5f} tot: {:.5f} ".format(result_stats[0], result_stats[1], result_stats[2]))


def benchmark_model(config, folder_model, batch_size = 1):
    print('Using GPUs {}'.format(config["gpu"]))
    device = torch.device('cuda:{}'.format(config["gpu"]))
    
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
                                seq_len=6,
                                randomize_seq=True)
    test_set = obd.dataset.BDD(root='/home/lecampos/data/bdd100k', dataset='track',
                                train=False, 
                                seq_len=6,
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
    #loss, accuracy = test(net, test_loader)
    #print("Float " + config["model"] + "_" + config['model_type'] +" loss: {:.5f} accuracy: {:.5f}".format(loss, accuracy))
 
    cudnn.benchmark = True
    #print("Float " + config["model"] + "_" + config['model_type'] +" time")
    #benchmark(net, test_loader, batch_size)

    qat_model = Network(num_classes=11, 
                      yolo_type=config['model_type'],
                      anchors=yolo_anchors, quantize=True).to(device)
    qat_model = qat_model.cuda()
    qat_model.init_model((448, 448, 3))
    
    qat_model.load_state_dict(torch.load(folder_model + '/network.pt', 
                                      map_location= 'cuda:{}'.format(config["gpu"])))
    
    #calibrate if model not found
    #if not os.path.exists(folder_model + "/" + config["model"] + "_" + config['model_type']+ "_qat.jit.pt"):
    #Calibrate the model using max calibration technique.
    with torch.no_grad():
        calibrate_model(
            model=qat_model,
            model_name=config["model"] + '_' + config['model_type'],
            data_loader=train_loader,
            num_calib_batch=int(0.001*len(train_loader)),
            calibrator="max",
            hist_percentile=[99.9, 99.99, 99.999, 99.9999],
            out_dir=folder_model + '/')
        
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    with torch.no_grad():
        data_in = iter(test_loader)
        images_raw, _, _ = next(data_in)
        images = images_raw[...,0]
        jit_model = torch.jit.trace(qat_model, images.to("cuda"))
        torch.jit.save(jit_model, folder_model + "/" + config["model"] + "_" + config['model_type']+ "_qat.jit.pt")

    qat_model = torch.jit.load(folder_model + "/" + config["model"] + "_" + config['model_type']+ "_qat.jit.pt").eval()
    B, C, W, H = images.shape
    compile_spec = {"inputs": [torch_tensorrt.Input([B, C, W, H])],
                    "enabled_precisions": torch.int8,
                    "truncate_long_and_double": True,
                    }
    trt_mod = torch_tensorrt.compile(qat_model, **compile_spec)

    loss, accuracy = test(jit_model, test_loader)
    print("jit_model loss: {:.5f} accuracy: {:.5f}".format(loss, accuracy))

    loss, accuracy = test(trt_mod, test_loader)
    print("trt_mod loss: {:.5f} accuracy: {:.5f}".format(loss, accuracy))
    
    print("jit_model time")
    benchmark(jit_model, test_loader, batch_size)
    print("trt_mod time")
    benchmark(trt_mod, test_loader, batch_size)
    

if __name__ == '__main__':
    config = {
        'gpu': 0,
        # Model
        'model': 'tiny_yolov3_ann', # yolov3_ann tiny_yolov3_ann
        'model_type': 'single-head', # complete, str, single-head
        # Target generation
        'tgt_iou_thr': 0.5,
    }
    base_path = '/home/lecampos/elrond91/lava-dl/tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/models/'
    benchmark_model(config, base_path + 'Trained_tiny_yolov3_ann_single-head')
   

    
    