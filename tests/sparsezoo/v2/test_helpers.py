import pytest
from sparsezoo import Zoo
from sparsezoo.v2.helpers import restructure_api_input
from sparsezoo.requests.download import download_model_get_request

NLP_TRAINING_FILES = {"trainer_state.json", "config.json", "special_tokens_map.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "training_args.bin", "pytorch_model.bin"}
NLP_DEPLOYMENT_FILES = NLP_TRAINING_FILES | {"model.onnx"}
NLP_FILE_NAMES = {'deployment': NLP_DEPLOYMENT_FILES,
                  'training': NLP_TRAINING_FILES,
                  'outputs': {"sample_outputs_onnxruntime.tar.gz", "sample_outputs_deepsparse.tar.gz"},
                  'onnx': {"model.onnx", "model.11.onnx", "model.14.onnx"},
                  'recipe': {"recipe_foo.md", "recipe_original.md"},
                  'card': {"model.md"},
                  'benchmarking': {"benchmarks.yaml", "eval.yaml", "analysis.yaml"},
                  'onnx_gz': {"model.onnx.tar.gz"},
                  'labels': {"sample_labels.tar.gz"},
                  'originals': {"sample_originals.tar.gz"},
                  'inputs': {"sample_inputs.tar.gz"},
                  'tar_gz': {"model.tar.gz"}
                  }

CV_TRAINING_FILES = {"model.pt", "model.ckpt.pt"}
CV_DEPLOYMENT_FILES = CV_TRAINING_FILES | {"model.onnx"}
CV_FILE_NAMES = {'deployment': CV_DEPLOYMENT_FILES,
                  'training': CV_TRAINING_FILES,
                  'outputs': {"sample_outputs.tar.gz"},
                  'onnx': {"model.onnx", "model.11.onnx", "model.14.onnx"},
                  'recipe': {"recipe_transfer_learn.md", "recipe_original.md"},
                  'card': {"model.md"},
                  'benchmarking': {"benchmarks.yaml", "eval.yaml", "analysis.yaml"},
                  'onnx_gz': {"model.onnx.tar.gz"},
                  'labels': {"sample_labels.tar.gz"},
                  'originals': {"sample_originals.tar.gz"},
                  'inputs': {"sample_inputs.tar.gz"},
                  'tar_gz': {"model.tar.gz"}
                  }


@pytest.mark.parametrize("model_stub, expected_file_names, runtime_specific_outputs",
                                                            [
                                                             ("zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95", NLP_FILE_NAMES, True),
                                                             ("zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94", CV_FILE_NAMES, False)
                                                             ])
def test_manager_staged_recipe_serialization(model_stub, expected_file_names, runtime_specific_outputs):
    model = Zoo.load_model_from_stub(model_stub)
    request_json = download_model_get_request(args=model)["model"]["files"]
    request_json = restructure_api_input(request_json, runtime_specific_outputs=runtime_specific_outputs)
    file_names = {(file_dict['display_name'], file_dict['file_type']) for file_dict in request_json}
    for file_type, files in expected_file_names.items():
        for display_name in files:
            file_names.remove((display_name, file_type))

    assert not file_names
