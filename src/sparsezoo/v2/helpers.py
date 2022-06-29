from sparsezoo.requests.download import download_model_get_request
from sparsezoo import Zoo
from typing import List, Dict, Any
import copy

def _fetch_from_request_json(request_json, key, value):
    return [(idx, copy.copy(file_dict)) for (idx, file_dict) in enumerate(request_json) if file_dict[key] == value]

def restructure_api_input(request_json: List[Dict[str, Any]], runtime_specific_outputs: bool = True):

    # convert `framework` to `training`
    # also create part of the `deployment` folder
    data = _fetch_from_request_json(request_json, 'file_type', 'framework')
    for (idx, file_dict) in data:
        file_dict['file_type'] = 'training'
        request_json[idx] = file_dict

        deployment_file_dict = copy.copy(file_dict)
        file_dict['file_type'] = 'deployment'
        request_json.append(deployment_file_dict)

    # finish the creation of the `deployment` folder
    data = _fetch_from_request_json(request_json, 'display_name', 'model.onnx')
    assert len(data) == 1
    _, file_dict = data[0]
    file_dict['file_type'] = 'deployment'
    request_json.append(file_dict)

    # restructure recipes
    data = _fetch_from_request_json(request_json, 'file_type', 'recipe')
    only_one_recipe = len(data) == 1
    for (idx, file_dict) in data:
        display_name = file_dict['display_name']
        if not display_name.startswith('recipe'):
            display_name = 'recipe_' + display_name
            file_dict['display_name'] = display_name
            request_json[idx] = file_dict
    if only_one_recipe:
        additional_file_dict = copy.copy(file_dict)
        additional_file_dict['display_name'] = "recipe_foo.md"
        request_json.append(additional_file_dict)

    # create yaml files
    # use model card to simulate them
    files_to_create = ["analysis.yaml", "benchmarks.yaml", "eval.yaml",]
    data = _fetch_from_request_json(request_json, 'display_name', 'model.md')
    assert len(data) == 1
    for file_name in files_to_create:
        file_dict = copy.copy(data[0][1])
        file_dict['display_name'] = file_name
        file_dict['file_type'] = 'benchmarking'
        request_json.append(file_dict)

    # create files for onnx directory
    files_to_create = ["model.11.onnx", "model.14.onnx",]
    data = _fetch_from_request_json(request_json,'file_type', 'onnx')
    assert len(data) == 1
    for file in files_to_create:
        file_dict = copy.copy(data[0][1])
        file_dict['display_name'] = file
        file_dict['operator_version'] = int(file.split(".")[-2])
        request_json.append(file_dict)

    # create logs folder
    data = _fetch_from_request_json(request_json,'display_name', 'sample-inputs.tar.gz')
    assert len(data) == 1
    _, file_dict = data[0]
    file_dict['display_name'] = 'logs.tar.gz'
    file_dict['file_type'] = 'logs'

    # numpy dirs
    _data = _fetch_from_request_json(request_json, 'display_name', 'sample-inputs.tar.gz')
    files_to_create = ["sample_inputs.tar.gz", "sample_labels.tar.gz", "sample_originals.tar.gz", "sample_outputs.tar.gz"]
    types = ["inputs", "labels", "originals", "outputs"]
    for file, type in zip(files_to_create, types):
        data = _fetch_from_request_json(request_json,'display_name', file.replace("_", "-"))
        if len(data) == 0:
            file_dict= copy.copy(_data[0][1])
            file_dict['display_name'] = file
            file_dict['file_type'] = type
            request_json.append(file_dict)
        elif len(data) == 1:
            file_dict= copy.copy(_data[0][1])
            file_dict['display_name'] = file
            file_dict['file_type'] = type
            idx = data[0][0]
            request_json[idx] = file_dict
        else:
            raise ValueError("")

    if runtime_specific_outputs:
        files_to_create = ["sample_outputs_deepsparse.tar.gz", "sample_outputs_onnxruntime.tar.gz"]
        data = _fetch_from_request_json(request_json, 'display_name', "sample_outputs.tar.gz")
        assert len(data) == 1
        for file in files_to_create:
            idx, file_dict = copy.copy(data[0])
            file_dict['display_name'] = file
            request_json.append(copy.copy(file_dict))
        del request_json[idx]

    return request_json



domain, sub_domain = 'nlp', 'question_answering'
model = Zoo.search_models(domain=domain, sub_domain=sub_domain)[0]
request_json = download_model_get_request(args=model)["model"]["files"]
restructure_api_input(request_json)