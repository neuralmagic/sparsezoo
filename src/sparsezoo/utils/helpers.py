"""
Code related to helper functions for model zoo
"""

import errno
import os


__all__ = [
    "BASE_API_URL",
    "CACHE_DIR",
    "clean_path",
    "create_dirs",
    "create_parent_dirs",
]

BASE_API_URL = "https://api.neuralmagic.com/models/"

CACHE_DIR = os.path.expanduser(os.path.join("~", ".cache", "nm_models"))


def clean_path(path: str) -> str:
    """
    :param path: the directory or file path to clean
    :return: a cleaned version that expands the user path and creates an absolute path
    """
    return os.path.abspath(os.path.expanduser(path))


def create_dirs(path: str):
    """
    :param path: the directory path to try and create
    """
    path = clean_path(path)

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            # Unexpected OSError, re-raise.
            raise


def create_parent_dirs(path: str):
    """
    :param path: the file path to try to create the parent directories for
    """
    parent = os.path.dirname(path)
    create_dirs(parent)
