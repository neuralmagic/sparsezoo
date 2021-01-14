import os

from sparsezoo.objects import Model


def validate_model_downloaded(
    model: Model, check_model_args=None, check_other_args=None
):
    if check_model_args:
        for key, value in check_model_args.items():
            assert getattr(model, key) == value

    if check_other_args:
        if "override_parent_path" in check_other_args:
            assert check_other_args["override_parent_path"] in model.dir_path
        if "override_folder_name" in check_other_args:
            assert check_other_args["override_folder_name"] in model.dir_path

    assert os.path.exists(model.dir_path)
    assert os.path.exists(model.card_file.path)
    assert os.path.exists(model.onnx_file.path)

    assert len(model.framework_files) > 0
    for file in model.framework_files:
        assert os.path.exists(file.path)

    assert len(model.recipes) > (0 if model.optim_name != "base" else -1)
    for recipe in model.recipes:
        assert os.path.exists(recipe.path)

    assert os.path.exists(model.data_inputs.path)
    assert os.path.exists(model.data_outputs.path)

    num_batches = 0
    for batch in model.loader(batch_size=16, iter_steps=5):
        assert "inputs" in batch
        assert "outputs" in batch
        num_batches += 1
    assert num_batches == 5
