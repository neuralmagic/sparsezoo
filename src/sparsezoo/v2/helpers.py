"""
Helper functions pertaining to the creation of ModelDirectory
"""

import copy
from pathlib import Path
from sparsezoo.v2 import File, Directory, NumpyDirectory, SampleOriginals
from typing import Union, List
import logging
import os
import shutil

def setup_model_directory(
        output_dir: str,
        training: Union[str, Directory],
        deployment: Union[str, Directory],
        onnx_model: Union[File, str],
        sample_inputs: Union[str, NumpyDirectory],
        sample_outputs: Union[str, NumpyDirectory, None] = None,
        sample_labels: Union[Directory, str, None] = None,
        sample_originals: Union[SampleOriginals, str, None] = None,
        logs: Union[Directory, str, None] = None,
        analysis: Union[File, str, None] = None,
        benchmarks: Union[File, str, None] = None,
        eval_results: Union[File, str, None] = None,
        model_card: Union[File, str, None] = None,
        recipes: Union[List[any], str, File, None] = None,
):
        """
        The function takes Files and Directories that are expected by the
        ModelDirectory (some Files/Directories are mandatory, some are optional),
        and then create a new Directory and copy in all those files into the correct place
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        files_dict = copy.deepcopy(locals())
        del files_dict['output_dir']
        for name, file in files_dict.items():
                if file is None:
                        logging.debug(f"File {name} not provided, skipping...")
                else:
                        _copy_file_contents(name, file, output_dir)

def _copy_file_contents(name, file, output_dir):
        is_dir = False
        if isinstance(file, str):
                path = file
                is_dir = os.path.isdir(path)
        elif isinstance(file, list):
                for _file in file:
                        if isinstance(_file, str):
                                file = _file
                                name = os.path.basename(_file)
                        else:
                                name = _file.name
                        _copy_file_contents(name = name, file=_file, output_dir=output_dir)
                        return
        elif isinstance(file, Directory):
                path = file.path
                is_dir = True
        else:
                path = file.path

        if name in ['training', 'deployment', 'logs']:
                copy_path = os.path.join(output_dir, name)
        else:
                copy_path = os.path.join(output_dir, os.path.basename(path))
        if is_dir:
                shutil.copytree(path, copy_path)
        else:
                shutil.copyfile(path, copy_path)






