
from sparsezoo import Zoo

from sparsezoo.analysis import ModelAnalysis
from sparsezoo.analysis.utils.chart import (
    draw_param_sparsity_chart
)

if __name__ == "__main__":
    #zoo_model = Zoo.load_model_from_stub("zoo:cv/classification/resnet_v1-50"
    #"/pytorch/sparseml/imagenet/pruned85_quant-none-vnni")
    zoo_model = Zoo.load_model_from_stub("zoo:nlp/question_answering/bert-base/"
    "pytorch/huggingface/squad/"
    "12layer_pruned80_quant-none-vnni")
    zoo_model.onnx_file.download()
    onnx_path = zoo_model.onnx_file.downloaded_path()

    analysis = ModelAnalysis.from_onnx_model(onnx_path)

    draw_param_sparsity_chart(analysis, "/Users/poketopa/Desktop/draw_param_sparsity_chart.png")
