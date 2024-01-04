# How to train and run Loihi (CPU) 2 Yolo for the Prophesee dataset

This repository demonstrates how to use Yolo SDNN Lava-dl Events for [GPU-based training](https://github.com/lava-nc/lava-dl/blob/demo/tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/train_sdnn_dvs.py) and [Loihi (CPU) 2 inference](https://github.com/lava-nc/lava-dl/blob/demo/tutorials/lava/lib/dl/netx/yolo_kp/run_dvs.ipynb). This tutorial specifies how to setup the environment for GPU inference, training and Loihi (CPU) 2 inference by three steps:
1. Setup [Prophesee Dataset](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) and [Toolbox](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/tree/master).
2. Train a Yolo Sigma Delta Neural Network (SDNN) model in Lava-dl on the Prophesee dataset
3. Run Yolo SDNN inference on Prophesee on Loihi (CPU) 2

Finally, if one desires a quick-start demonstration of Prophesee Loihi 2 inference from the training pipeline above, this repository also contains the resources to load a pre-trained SDNN Yolo Prophesee model (the output from step 2) and run inference on a subset of Prophesee samples (the output of step 1).

## Prerequisites
Install this respository, see top level README.md

## Step 1: Install Prophesee Dataset and Toolbox
The dataset is split between train, test and val folders. 

Files consist of 60 seconds recordings that were cut from longer recording sessions. Cuts from a single recording session are all in the same training split.

Each dat file is a binary file in which events are encoded using 4 bytes (unsigned int32) for the timestamps and 4 bytes (unsigned int32) for the data, encoding is little-endian ordering.

The data is composed of 14 bits for the x position, 14 bits for the y position and 1 bit for the polarity (encoded as -1/1).

Annotations use the numpy format and can simply be loaded form python using numpy boxes = np.load(path)

Boxes have the following fields

* x abscissa of the top left corner in pixels
* y ordinate of the top left corner in pixels
* w width of the boxes in pixel
* h height of the boxes in pixel
* ts timestamp of the box in the sequence in microseconds
* class_id 0 for pedestrians, 1 for two wheelers, 2 for cars, 3 for trucks, 4 for buses, 5 for traffic signs, 6 for traffic lights

Register at:

https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/

to download the zip folders; Then, clone the toolbox:

` git clone https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/tree/master`

and add it to python libraries path. 

## Step 2: Train Yolo SDNN on PropheseeAutomotive 
Requires NVIDIA GPU. In `https://github.com/lava-nc/lava-dl/blob/demo/tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/train_sdnn_dvs.py` has been enhanced with the `PropheseeAutomotive` dataset option.

Modifying paths appropriately, you can train on the PropheseeAutomotive dataset with a command like
`python train_sdnn_dvs.py -model yolo_kp_events -epoch 200 -lr 0.0001 -lrf 0.01 -warmup 40 -lambda_coord 2 -lambda_noobj 4 -lambda_obj 1.8 -lambda_cls 1 -lambda_iou 2.25 -alpha_iou 0.8 -clip 1 -label_smoothing 0.03 -tgt_iou_thr 0.25 -aug_prob 0.4 -track_iter 1  -sparsity -sp_lam 0.01 -sp_rate 0.01 -dataset PropheseeAutomotive -path /path/to/PropheseeAutomotive-directory -num_workers 2 -b 1 -output_dir ./run1`


## Step 3: Run Yolo SDNN inference for PropheseeAutomotive on Loihi (CPU) 2
In the "Set execution parameters" section of `lava-dl/tutorials/lava/lib/dl/netx/yolo_kp/run_dvs.ipynb`, you will see a comment that notes to modify the path for the trained model folder to point to your trained PropheseeAutomotive model folder from Step 2. Also check that `path` in `args.txt` in the trained model foler points to the location of the preprocessed PropheseeAutomotive dataset from Step 1.

Running this notebook now performs SDNN inference for PropheseeAutomotive on Loihi (CPU) 2.

## Quickstart demonstration: run inference on Loihi 2 using a pretrained PropheseeAutomotive model 
Skip Steps 1 and 2. 

Extract the sample PropheseeAutomotive dataset `ai.ncl.afrl-demo/datasets/prophesee/prophesee_sample.tar.gz`. (This sample is only a portion of the full PropheseeAutomotive dataset.)

Perform Step 3, except in `lava-dl/tutorials/lava/lib/dl/netx/yolo_kp/run_dvs.ipynb` ensure the trained folder path points to the pretrained model that comes with this repository (`ai.ncl.afrl-demo/datasets/prophesee/Trained_yolo_kp_dvs`), and set `path` in `args.txt` in this folder to the location at which you have extracted the sample prophesee dataset.

Running `lava-dl/tutorials/lava/lib/dl/netx/yolo_kp/run_dvs.ipynb` now performs SDNN inference for prophesee on Loihi (CPU) 2 using the pretrained model, and pulls input from the sample prophesee dataset.