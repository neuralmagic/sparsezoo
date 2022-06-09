
from sparsezoo import Zoo

from sparsezoo.analysis import ModelAnalysis
from sparsezoo.analysis.utils.chart import (
    draw_param_sparsity_chart
)

if __name__ == "__main__":
    model_stubs = {
        #"yolact_none": "zoo:cv/segmentation/yolact-darknet53/"
        #"pytorch/dbolya/coco/base-none",
        #"mobilenet_v1_pruned_moderate": "zoo:cv/classification/mobilenet_v1-1.0/"
        #"pytorch/sparseml/imagenet/pruned-moderate",
        #"bert_pruned_quantized": "zoo:nlp/question_answering/bert-base/"
        #"pytorch/huggingface/squad/"
        #"12layer_pruned80_quant-none-vnni",
        "resnet50_pruned_quantized": "zoo:cv/classification/resnet_v1-50"
        "/pytorch/sparseml/imagenet/pruned85_quant-none-vnni",
    }

    for model_name, model_stub in model_stubs.items():
        zoo_model = Zoo.load_model_from_stub(model_stub)
        zoo_model.onnx_file.download()
        onnx_path = zoo_model.onnx_file.downloaded_path()

        analysis = ModelAnalysis.from_onnx_model(onnx_path)

        draw_param_sparsity_chart(analysis, "/Users/poketopa/Desktop/draw_param_sparsity_chart.png", model_name=model_name)
