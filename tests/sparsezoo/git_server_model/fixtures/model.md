---
card_version: 1.0.0
domain: cv
base: _some_base_stub
task: classification
architecture: resnet_v1
sub_architecture: 50
framework: pytorch
repo: sparseml
version: null
source_dataset: imagenet
train_dataset: imagenet_2
display_name: 95% Pruned ResNet-50
tags:
- resnet
- resnet_v1
- resnet50
- pruned
- pruned95
- sparseml
- pytorch
- imagenet
parent: _some_parent_stub
optimizations: 
- GMP 95%
- QAT Int8
commands:
  deploy:
    command1: sparseml.command.dosomething stub
    deploy_model: deepsparse.command.dosomething stub
  train:
    command3: sparseml.command.foo
    train_model: sparseml.command.bar stub
    train_model_stop_at_epoch_20: python3 sparseml.command.bar
  benchmark: 
    benchmark: deepsparse.benchmark stub
    benchmark_on_instance: deepsparse.benchmark --instance_type c5.12xlarge --stub zoo_stub


---

# ResNet-50

This model results from pruning 95% of parameters of the [ResNet-50](https://arxiv.org/abs/1512.03385) model from [Torchvision](https://pytorch.org/vision/stable/models.html) trained on the [Imagenet 2012 dataset](https://image-net.org/challenges/LSVRC/2012/).
This model achieves 75.9% top1 validation accuracy, recovering over 99% of the top1 validation accuracy of baseline model (76.1%).

## Training

A sample command for creating this model on ImageNet (edit for training environment and dataset):

```bash
sparseml.image_classification.train \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none?recipe_type=original \
    --pretrained True \
    --arch-key resnet50 \
    --dataset imagenet \
    --dataset-path /PATH/TO/IMAGENET  \
    --train-batch-size 320 --test-batch-size 1000 \
    --loader-num-workers 16 \
    --model-tag resnet50-imagenet-pruned95-none
```