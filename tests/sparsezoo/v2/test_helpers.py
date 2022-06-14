import tempfile
import pytest
from sparsezoo.v2 import Directory, NumpyDirectory, File
from tests.sparsezoo.v2.test_model_directory import TestModelDirectory
from sparsezoo.v2.helpers import setup_model_directory
from sparsezoo import Zoo
import glob
import os


@pytest.mark.parametrize(
    "stub",
    ["zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95"]
)

class TestSetupModelDirectory:
    @pytest.fixture()
    def setup(self, stub):
        # setup
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        output_dir = tempfile.TemporaryDirectory(dir="/tmp")
        model = Zoo.download_model_from_stub(stub, override_folder_name=temp_dir.name)

        yield model, temp_dir, output_dir
        temp_dir.cleanup()
        output_dir.cleanup()

    def test_setup_model_directory_from_paths(self, setup):
        model, temp_dir, output_dir = setup
        setup_model_directory(output_dir = output_dir.name,
                              training = model.framework_files[0].dir_path,
                              deployment = model.framework_files[0].dir_path,
                              onnx_model = model.onnx_file.path,
                              sample_inputs = model.data_inputs.path,
                              sample_outputs = model.data_outputs.path,
                              recipes = [model.recipes[0].path] * 2)

        folders = glob.glob(os.path.join(output_dir.name, "*"))
        assert {'training', 'deployment', 'original.md',
                'model.onnx', 'sample-outputs.tar.gz', 'sample-inputs.tar.gz'} == set(os.path.basename(file) for file in folders)


    def test_setup_model_directory(self, setup):
        model, temp_dir, output_dir = setup
        training_folder_path = model.framework_files[0].dir_path
        training = File(name=os.path.basename(training_folder_path), path = training_folder_path)
        training = Directory.from_file(file=training)

        deployment_folder_path = model.framework_files[0].dir_path
        deployment = File(name=os.path.basename(deployment_folder_path), path=deployment_folder_path)
        deployment = Directory.from_file(file=deployment)

        onnx_model = File(name=os.path.basename(model.onnx_file.path), path=model.onnx_file.path)

        sample_inputs = File(name=os.path.basename(model.data_inputs.path), path=model.data_inputs.path)

        sample_outputs = File(name=os.path.basename(model.data_outputs.path), path=model.data_outputs.path)

        recipes = [File(name=os.path.basename(model.recipes[0].path), path=model.recipes[0].path)] * 2


        setup_model_directory(output_dir=output_dir.name,
                              training=training,
                              deployment=deployment,
                              onnx_model=onnx_model,
                              sample_inputs=sample_inputs,
                              sample_outputs=sample_outputs,
                              recipes=recipes)
        folders = glob.glob(os.path.join(output_dir.name, "*"))
        assert {'training', 'deployment', 'original.md',
                'model.onnx', 'sample-outputs.tar.gz', 'sample-inputs.tar.gz'} == set(
            os.path.basename(file) for file in folders)


