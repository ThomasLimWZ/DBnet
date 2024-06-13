## Requirements:
Use T4/A100 runtime in Google Colab, because it requires GPU to run the code.
- Python3
- PyTorch == 1.2 
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (10.1 is recommended)

There are some files might need to modify the path to run the code (You may use `ctrl + F` to search "drive" as keyword to modify):
* `base/base_totaltext.yaml`
* `base/base_ic15.yaml`
* `base/base_ic17.yaml`

```bash
  # first, make sure that your conda is setup properly with the right environment

  # python dependencies
  !pip install pyyaml
  !pip install tqdm
  !pip install tensorboardX
  !pip install opencv-python
  !pip install anyconfig
  !pip install munch
  !pip install scipy
  !pip install sortedcontainers
  !pip install shapely==2.0.3
  !pip install pyclipper
  !pip install gevent
  !pip install gevent_websocket
  !pip install flask
  !pip install editdistance
  !pip install scikit-image
  !pip install imgaug==0.4
  !pip install easyocr
  !pip install numpy==1.23

  # build deformable convolution opertor
  # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
  # make sure GCC >= 4.9
  # you need to delete the build directory before you re-build it.
  echo $CUDA_HOME
  cd assets/ops/dcn/
  python setup.py build_ext --inplace
```

## SynthText Pre-trained model
```
├── trained_models
│   ├── pre-trained-model-synthtext-resnet50   -- used to finetune models, not for evaluation
```

## Datasets
The root of the dataset directory is in ```datasets/```.
```
├── datasets
│   ├── totaltext
│   │   ├── train_images
│   │   └── train_gts
│   │   ├── test_images
│   │   └── test_gts
│   │   └── test_list.txt
│   │   └── train_list.txt
│   ├── icdar2015
│   │   ├── train_images
│   │   └── train_gts
│   │   ├── test_images
│   │   └── test_gts
│   │   └── test_list.txt
│   │   └── train_list.txt
│   ├── icdar2017
│   │   ├── train_images
│   │   └── train_gts
│   │   ├── test_images
│   │   └── test_gts
│   │   └── test_list.txt
│   │   └── train_list.txt
```
The data root directory and the data list file can be defined in ```base_totaltext.yaml```

## Config file
**The YAML files with the name of ```base*.yaml``` should not be used as the training or testing config file directly.**

## Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.
During the training process, it will generates several items and save in ```outputs/workspace/DBnet/deformable_resnet50/L1BalanceCELoss``` or ```outputs/workspace/DBnet/resnet50/L1BalanceCELoss``` directory.
```
├── model      # Checkpoint stores every specific iterations
│   ├── final
│   ├── ...
├── summaries  # For tensorboardX visualize graph
│   ├── ...
├── visualize  # Visualize the bounding boxes on the images of testing set
│   ├── img_1.jpg_output.jpg
│   ├── ...
├── args.log   # A combined parameters settings from YAML files
├── epoch
├── iter
├── metrics.log
├── output.log  # Store the code ran previously
```

```
!python train.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml

!python train.py experiments/seg_detector/ic15_resnet50_thre.yaml --resume trained_models/pre-trained-model-synthtext-resnet50

!python train.py experiments/seg_detector/ic17_resnet50_deform_thre.yaml --resume trained_models/pre-trained-model-synthtext-resnet50
```

## View Graph using TensorboardX
```
%reload_ext tensorboard
%tensorboard --logdir=workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/totaltext/summaries

%reload_ext tensorboard
%tensorboard --logdir=workspace/SegDetectorModel-seg_detector/resnet50/L1BalanceCELoss/summaries

%reload_ext tensorboard
%tensorboard --logdir=workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/ic17/summaries
```

## Evaluate the performance
```
!python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume outputs/workspace/DBnet/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/totaltext/model/final --polygon --box_thresh 0.6 --visualize

!python eval.py experiments/seg_detector/ic15_resnet50_thre.yaml --resume outputs/workspace/DBnet/SegDetectorModel-seg_detector/resnet50/L1BalanceCELoss/model/final --polygon --box_thresh 0.7 --visualize

!python eval.py experiments/seg_detector/ic17_resnet50_deform_thre.yaml --resume outputs/workspace/DBnet/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/ic17/model/final --polygon --box_thresh 0.5 --visualize
```

```box_thresh``` can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. ```polygon``` is only used for arbitrary-shape text dataset. The size of the input images are defined in ```validate_data->processes->AugmentDetectionData``` in ```base_*.yaml```.

It also will generate the ground truth of the bounding boxes in text file for each image in testing set and stores in ```results/```.
The IoU of each image will be write in a text file named as ```iou.txt```. The average IoU also will be included at the last line.

## Evaluate the speed 
Set ```adaptive``` to ```False``` in the yaml file to speedup the inference without decreasing the performance. The speed is evaluated by performing a testing image for 50 times to exclude extra IO time.

```!python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/totaltext/model/final --polygon --box_thresh 0.6 --speed```

Note that the speed is related to both to the GPU and the CPU since the model runs with the GPU and the post-processing algorithm runs with the CPU.

## Demo
Run the model inference with a single image. Here is an example:
```!python demo.py experiments/seg_detector/ic17_resnet50_deform_thre.yaml --image_path images/starbucks.jpg --resume workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/ic17/model/final --box_thresh 0.55 --polygon --visualize```

The results can be find in `demo_results`.
```
├── starbucks.jpg
├── res_starbucks.txt
├── starbucks_0.jpg  # Extract out each detected text before bounding box generated
├── starbucks_1.jpg
├── recognized.txt   # Stores the text recognized by EasyOCR library with the probabilty
│   ├── starbucks_0.jpg ~ Text: CNFE, Probability: 0.09245385229587555
│   ├── starbucks_1.jpg ~ Text: STAucrs, Probability: 0.0687920127407071
```
