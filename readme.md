# Yet Another EfficientDet Pytorch

The pytorch re-implement of the official [EfficientDet](https://github.com/google/automl/tree/master/efficientdet) with SOTA performance in real time, original paper link: <https://arxiv.org/abs/1911.09070>

## Having troubles training? I might train it for you

If you have troubles training a dataset, and if you are willing to share your dataset with the public or it's open already, post it on Issues with `help wanted` tag, I might try to help train it for you, if I'm free, which is not guaranteed.

Requirements:

1. The total number of the image of the dataset should not be larger than 10K, capacity should be under 5GB, and it should be free to download, i.e. baiduyun.

2. The dataset should be in the format of this repo.

3. If you post your dataset in this repo, it is open to the world. So PLEASE DO NOT upload your confidential datasets!

4. If the datasets are against the law or invade one's privacy, feel free to contact me to delete it.

5. Most importantly, you can't demand me to train unless I wanted to.

I'll post the trained weights in this repo along with the evaluation result.

Hope it help whoever wants to try efficientdet in pytorch.

Training examples can be found here. [tutorials](tutorial/). The trained weights can be found here. [weights](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/tag/custom_datasets)


## Performance

## Pretrained weights and benchmark

The performance is very close to the paper's, it is still SOTA.

The speed/FPS test includes the time of post-processing with no jit/data precision trick.

| coefficient | pth_download | GPU Mem(MB) | FPS | Extreme FPS (Batchsize 32) | mAP 0.5:0.95(this repo) | mAP 0.5:0.95(official) |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| D0 | [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) | 1049 | 36.20 | 163.14 | 33.1 | 33.8
| D1 | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth) | 1159 | 29.69 | 63.08 | 38.8 | 39.6
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 1321 | 26.50 | 40.99 | 42.1 | 43.0
| D3 | [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) | 1647 | 22.73 | - | 45.6 | 45.8
| D4 | [efficientdet-d4.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth) | 1903 | 14.75 | - | 48.8 | 49.4
| D5 | [efficientdet-d5.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth) | 2255 | 7.11 | - | 50.2 | 50.7
| D6 | [efficientdet-d6.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth) | 2985 | 5.30 | - | 50.7 | 51.7
| D7 | [efficientdet-d7.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d7.pth) | 3819 | 3.73 | - | 52.7 | 53.7
| D7X | [efficientdet-d8.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d8.pth) | 3983 | 2.39 | - | 53.9 | 55.1

## Update Log

[2020-07-23] supports efficientdet-d7x, mAP 53.9, using efficientnet-b7 as its backbone and an extra deeper pyramid level of BiFPN. For the sake of simplicity, let's call it efficientdet-d8.

[2020-07-15] update efficientdet-d7 weights, mAP 52.7

[2020-05-11] add boolean string conversion to make sure head_only works

[2020-05-10] replace nms with batched_nms to further improve mAP by 0.5~0.7, thanks [Laughing-q](https://github.com/Laughing-q).

[2020-05-04] fix coco category id mismatch bug, but it shouldn't affect training on custom dataset.

[2020-04-14] fixed loss function bug. please pull the latest code.

[2020-04-14] for those who needs help or can't get a good result after several epochs, check out this [tutorial](tutorial/train_shape.ipynb). You can run it on colab with GPU support.

[2020-04-10] warp the loss function within the training model, so that the memory usage will be balanced when training with multiple gpus, enabling training with bigger batchsize.

[2020-04-10] add D7 (D6 with larger input size and larger anchor scale) support and test its mAP

[2020-04-09] allow custom anchor scales and ratios

[2020-04-08] add D6 support and test its mAP

[2020-04-08] add training script and its doc; update eval script and simple inference script.

[2020-04-07] tested D0-D5 mAP, result seems nice, details can be found [here](benchmark/coco_eval_result)

[2020-04-07] fix anchors strategies.

[2020-04-06] adapt anchor strategies.

[2020-04-05] create this repository.

## Demo

    # install requirements
    pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0
     
    # run the simple inference script
    python efficientdet_test.py

## Training

Training EfficientDet is a painful and time-consuming task. You shouldn't expect to get a good result within a day or two. Please be patient.

Check out this [tutorial](tutorial/) if you are new to this. You can run it on colab with GPU support.

### 1. Prepare your dataset

    # your dataset structure should be like this
    datasets/
        -your_project_name/
            -train_set_name/
                -*.jpg
            -val_set_name/
                -*.jpg
            -annotations
                -instances_{train_set_name}.json
                -instances_{val_set_name}.json
    
    # for example, coco2017
    datasets/
        -coco2017/
            -train2017/
                -000000000001.jpg
                -000000000002.jpg
                -000000000003.jpg
            -val2017/
                -000000000004.jpg
                -000000000005.jpg
                -000000000006.jpg
            -annotations
                -instances_train2017.json
                -instances_val2017.json

### 2. Manual set project's specific parameters

    # create a yml file {your_project_name}.yml under 'projects'folder 
    # modify it following 'coco.yml'
     
    # for example
    project_name: coco
    train_set: train2017
    val_set: val2017
    num_gpus: 4  # 0 means using cpu, 1-N means using gpus 
    
    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    
    # this is coco anchors, change it if necessary
    anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
    anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    
    # objects from all labels from your dataset with the order from your annotations.
    # its index must match your dataset's category_id.
    # category_id is one_indexed,
    # for example, index of 'car' here is 2, while category_id of is 3
    obj_list: ['person', 'bicycle', 'car', ...]

### 3.a. Train on coco from scratch(not necessary)

    # train efficientdet-d0 on coco from scratch 
    # with batchsize 12
    # This takes time and requires change 
    # of hyperparameters every few hours.
    # If you have months to kill, do it. 
    # It's not like someone going to achieve
    # better score than the one in the paper.
    # The first few epoches will be rather unstable,
    # it's quite normal when you train from scratch.
    
    python train.py -c 0 --batch_size 64 --optim sgd --lr 8e-2

### 3.b. Train a custom dataset from scratch

    # train efficientdet-d1 on a custom dataset 
    # with batchsize 8 and learning rate 1e-5
    
    python train.py -c 1 -p your_project_name --batch_size 8 --lr 1e-5

### 3.c. Train a custom dataset with pretrained weights (Highly Recommended)

    # train efficientdet-d2 on a custom dataset with pretrained weights
    # with batchsize 8 and learning rate 1e-3 for 10 epoches
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth
    
    # with a coco-pretrained, you can even freeze the backbone and train heads only
    # to speed up training and help convergence.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --head_only True

### 4. Early stopping a training session

    # while training, press Ctrl+c, the program will catch KeyboardInterrupt
    # and stop training, save current checkpoint.

### 5. Resume training

    # let says you started a training session like this.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --head_only True
     
    # then you stopped it with a Ctrl+c, it exited with a checkpoint
    
    # now you want to resume training from the last checkpoint
    # simply set load_weights to 'last'
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights last \
     --head_only True

### 6. Evaluate model performance

    # eval on your_project, efficientdet-d5
    
    python coco_eval.py -p your_project_name -c 5 \
     -w /path/to/your/weights

### 7. Debug training (optional)

    # when you get bad result, you need to debug the training result.
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --debug True
    
    # then checkout test/ folder, there you can visualize the predicted boxes during training
    # don't panic if you see countless of error boxes, it happens when the training is at early stage.
    # But if you still can't see a normal box after several epoches, not even one in all image,
    # then it's possible that either the anchors config is inappropriate or the ground truth is corrupted.

## TODO

- [X] re-implement efficientdet
- [X] adapt anchor strategies
- [X] mAP tests
- [X] training-scripts
- [X] efficientdet D6 support
- [X] efficientdet D7 support
- [X] efficientdet D7x support



