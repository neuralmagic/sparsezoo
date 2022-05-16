# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pathlib
import tarfile
from typing import List, Optional

from src.sparsezoo.refactor.file import File
from src.sparsezoo.utils.downloader import download_file


__all__ = ["Directory"]


class Directory(File):
    """
    Object that represents a directory.
    Directory may also represent a tar file.

    :param files: list of files contained within the Directory
        (None for tar files).
    :param name: name of the Directory
    :param path: path of the Directory
    :param url: url of the Directory
    """

    def __init__(
        self,
        name: str,
        files: Optional[List[File]] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):

        self.files = files

        super().__init__(name=name, path=path, url=url)

    def validate(self, strict_mode: bool = True):
        """
        Validates the structure of directory.

        :return: boolean flag; True if files are valid and no errors arise
        """
        validations = {}
        if self._is_tar():
            # we are not validating tar files for now
            validations[self.name] = True

        else:
            validations = {}
            for file in self.files:
                if isinstance(file, Directory):
                    # we are not validating nested directories for now
                    raise NotImplementedError(
                        "Method validate() does not support nested directories yet."
                    )
                else:
                    validations[file.name] = file.validate(strict_mode=strict_mode)

        if not all(validations.values()):
            logging.warning(
                "Following files: "
                f"{[key for key, value in validations.items() if not value]} "
                "were not successfully validated."
            )

        return all(validations.values())

    def download(self, destination_path: str, overwrite: bool = True):
        """
        Download the contents of the file from the url.

        :param destination_path: the local file path to save the downloaded file to
        :param overwrite: True to overwrite any previous files if they exist,
            False to not overwrite and raise an error if a file exists
        """
        # Directory can represent a tar file.
        if self._is_tar():
            download_file(
                url_path=self.url,
                dest_path=os.path.join(destination_path, self.name),
                overwrite=overwrite,
            )
            self.path = os.path.join(destination_path, self.name)

        # Directory can represent a folder.
        else:
            for file in self.files:
                if isinstance(file, Directory):
                    raise NotImplementedError(
                        "Method download() does not support nested directories yet."
                    )
                else:
                    file.download(destination_path=destination_path)

    @classmethod
    def gzip(
        cls, directory: "Directory", archive_directory: Optional[str] = None
    ) -> "Directory":
        """
        Factory method that creates a tar archive Directory
        from the `directory` (folder Directory).
        The tar archive file would be saved in the parent
        directory of the `directory`, unless `extract_directory` argument is specified.

        :param directory: the folder directory (Directory class object)
        :param archive_directory: the local path to
            save new tar archive Directory to (default = None)
        :return: tar archive Directory
        """
        if directory._is_tar():
            raise ValueError(
                "Attempting to create a tar archive "
                "Directory from a tar archive Directory."
            )
        if archive_directory is not None:
            parent_path = archive_directory
        else:
            if directory.path is None:
                raise ValueError(
                    "Attempting to zip the folder Directory object files using "
                    "`path` attribute, but `self.path` is None. "
                    "Class object requires pointer to parent "
                    "folder directory to know where to save the tar archive file."
                )
            parent_path = pathlib.PurePath(directory.path).parent

        tar_file_name = directory.name + ".tar.gz"
        tar_file_path = os.path.join(parent_path, tar_file_name)
        with tarfile.open(tar_file_path, "w") as tar:
            for file in directory.files:
                tar.add(file.path)
        return Directory(name=tar_file_name, path=tar_file_path)

    @classmethod
    def unzip(
        cls, tar_directory: "Directory", extract_directory: Optional[str] = None
    ) -> "Directory":
        """
        Factory method that extracts a tar archive `tar_directory` into a new
        folder Directory.
        The extracted files would be saved in the parent directory of
        the `tar_directory`, unless `extract_directory` argument is specified.

        :param tar_directory: the tar archive directory (Directory class object)
        :param extract_directory: the local path to create
            folder Directory at (and thus save extracted files at)
        :return: folder Directory
        """
        files = []
        if extract_directory is None:
            extract_directory = os.path.dirname(tar_directory.path)

        if not tar_directory._is_tar():
            raise ValueError(
                "Attempting to extract tar archive, "
                "but the Directory object is not tar archive"
            )

        name = os.path.basename(extract_directory)
        tar = tarfile.open(tar_directory.path, "r")
        path = extract_directory
        for member in tar.getmembers():
            member.name = os.path.basename(member.name)
            tar.extract(member=member, path=extract_directory)
            files.append(
                File(
                    name=member.name, path=os.path.join(extract_directory, member.name)
                )
            )
        tar.close()

        return Directory(name=name, files=files, path=path)

    def __len__(self):
        return len(self.files)

    def _is_tar(self):
        _, *extension = self.name.split(".")
        return (extension == ["tar", "gz"]) and not self.files
