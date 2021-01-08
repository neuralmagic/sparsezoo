# Available Models
A number of pre-trained models are available in this API. Included are both baseline and recalibrated models for higher performance. 
The types available for each model architecture are noted in the table below.

Possible types are:
 - base - the baseline model (standard training process)
 - recal - a recalibrated model for better performance that achieves ~100% of baseline validation metrics
 - recal-perf - a recalibrated model for better performance that meets ~99% of baseline validation metrics


|  Architecture       | Dataset  | Available Types         | Frameworks                 | Validation Baseline Metric |
| ------------------- | -------- | ----------------------- | -------------------------- | -------------------------- |
| MnistNet            | MNIST    | base                    | ONNX, PyTorch, TensorFlow  | ~99% top1 accuracy         |
| EfficientNet-B0     | ImageNet | base, recal-perf        | ONNX, PyTorch              | 77.3% top1 accuracy        |
| EfficientNet-B4     | ImageNet | base, recal-perf        | ONNX, PyTorch              | 83.0% top1 accuracy        |
| InceptionV3         | ImageNet | base, recal, recal-perf | ONNX, PyTorch              | 77.45% top1 accuracy       |
| MobileNetV1         | ImageNet | base, recal, recal-perf | ONNX, PyTorch, TensorFlow  | 70.9% top1 accuracy        |
| MobileNetV2         | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 71.88% top1 accuracy       |
| ResNet-18           | ImageNet | base, recal             | ONNX, PyTorch, TensorFlow  | 69.8% top1 accuracy        |
| ResNet-34           | ImageNet | base, recal             | ONNX, PyTorch, TensorFlow  | 73.3% top1 accuracy        |
| ResNet-50           | ImageNet | base, recal, recal-perf | ONNX, PyTorch, TensorFlow  | 76.1% top1 accuracy        |
| ResNet-50 2xwidth   | ImageNet | base                    | ONNX, PyTorch              | 78.51% top1 accuracy       |
| ResNet-101          | ImageNet | base, recal-perf        | ONNX, PyTorch, TensorFlow  | 77.37% top1 accuracy       |
| ResNet-101 2xwidth  | ImageNet | base                    | ONNX, PyTorch              | 78.84% top1 accuracy       |
| ResNet-152          | ImageNet | base, recal-perf        | ONNX, PyTorch, TensorFlow  | 78.31% top1 accuracy       |
| VGG-11              | ImageNet | base, recal-perf        | ONNX, PyTorch, TensorFlow  | 69.02% top1 accuracy       |
| VGG-11bn            | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 70.38% top1 accuracy       |
| VGG-13              | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 69.93% top1 accuracy       |
| VGG-13bn            | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 71.55% top1 accuracy       |
| VGG-16              | ImageNet | base, recal, recal-perf | ONNX, PyTorch, TensorFlow  | 71.59% top1 accuracy       |
| VGG-16bn            | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 71.55% top1 accuracy       |
| VGG-19              | ImageNet | base, recal-perf        | ONNX, PyTorch, TensorFlow  | 72.38% top1 accuracy       |
| VGG-19bn            | ImageNet | base                    | ONNX, PyTorch, TensorFlow  | 74.24% top1 accuracy       |
| SSD-300-ResNet-50   | COCO     | base, recal-perf        | ONNX, PyTorch              | 42.7% mAP@0.5              |
| SSD-300-ResNet-50   | VOC      | base, recal-perf        | ONNX, PyTorch              | 52.2% mAP@0.5              |
| SSDLite-MobileNetV2 | COCO     | base                    | ONNX, PyTorch              | 35.7% mAP@0.5              |
| SSDLite-MobileNetV2 | VOC      | base                    | ONNX, PyTorch              | 43.5% mAP@0.5              |
| YOLOv3              | COCO     | base, recal-perf        | ONNX, PyTorch              | 68.6% mAP@0.5              |