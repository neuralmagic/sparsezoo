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
import time
import traceback
from typing import List, Optional

from sparsezoo.utils.downloader import download_file
from sparsezoo.v2.file import File


__all__ = ["Directory"]


class Directory(File):
    """
    Object that represents a directory.
    Directory may also represent a tar file.

    :param files: list of files contained within the Directory
        (None for tar archive files)
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
        extension = name.split(".")[-2:]
        self._is_archive = (extension == ["tar", "gz"]) and (not self.files)

        super().__init__(name=name, path=path, url=url)

        if self._unpack():
            self.unzip()

    @property
    def is_archive(self) -> bool:
        """
        Boolean flag:
        - True if the Directory is an archive (tar)
        - False if the Directory is a local directory (folder)

        :return: boolean flag; True if Directory is tar archive, False if folder
        """
        return self._is_archive

    @is_archive.setter
    def is_archive(self, value: bool):
        """
        Setter property of `is_archive`

        :param value: boolean flag; True if Directory is tar archive, False if folder
        """
        self._is_archive = value

    def get_file_names(self) -> List[str]:
        """
        Get the names of the files in the Directory

        return: List with names of files
        """
        if self.is_archive:
            tar = tarfile.open(self.path)
            return [os.path.basename(member.name) for member in tar.getmembers()]
        else:
            return [file.name for file in self.files]

    def validate(self, strict_mode: bool = True):
        """
        Validates the structure of directory.

        :return: boolean flag; True if files are valid and no errors arise
        """

        if self.is_archive:
            # we are not validating tar files for now
            validations = {self.name: True}
        else:
            validations = {}
            for file in self.files:
                validations[file.name] = file.validate(strict_mode=strict_mode)

        if not all(validations.values()):
            logging.warning(
                "Following files: "
                f"{[key for key, value in validations.items() if not value]} "
                "were not successfully validated."
            )

        return all(validations.values())

    def download(
        self,
        destination_path: str,
        overwrite: bool = True,
        retries: int = 1,
        retry_sleep_sec: int = 5,
    ):
        """
        Download the contents of the file from the url

        :param destination_path: the local file path to save the downloaded file to
        :param overwrite: True to overwrite any previous files if they exist,
            False to not overwrite and raise an error if a file exists
        :param retries: The maximum number of times to ping the API for the response
        :type retry_sleep_sec: How long to wait between `retry` attempts
        """
        # Directory can represent a tar file.
        if self.is_archive:
            new_file_path = os.path.join(destination_path, self.name)
            for attempt in range(retries):
                try:
                    download_file(
                        url_path=self.url,
                        dest_path=new_file_path,
                        overwrite=overwrite,
                    )

                    self.path = new_file_path
                    return

                except Exception as err:
                    logging.error(err)
                    logging.error(traceback.format_exc())
                    time.sleep(retry_sleep_sec)
                logging.error(
                    f"Trying attempt {attempt + 1} of {retries}.", attempt + 1, retries
                )
            logging.error("Download retry failed...")
            raise Exception("Exceed max retry attempts: {} failed".format(retries))

        # Directory can represent a folder/directory.
        else:
            for file in self.files:
                file.download(destination_path=destination_path)

    def gzip(self, archive_directory: Optional[str] = None):
        """
        Creates a tar archive directory from the current directory.
        The resulting tar archive directory would be created in the parent
        directory of `self`, unless `extract_directory` argument is specified.

        :param archive_directory: the local path to
            save new tar archive Directory to (default = None)
        """
        if self.is_archive:
            raise ValueError(
                "Attempting to create a tar archive "
                "Directory from a tar archive Directory."
            )
        if archive_directory is not None:
            parent_path = archive_directory

        else:
            if self.path is None:
                raise ValueError(
                    "Attempting to zip the folder Directory object files using "
                    "`path` attribute, but `self.path` is None. "
                    "Class object requires pointer to parent "
                    "folder directory to know where to save the tar archive file."
                )
            parent_path = pathlib.PurePath(self.path).parent

        tar_file_name = self.name + ".tar.gz"
        tar_file_path = os.path.join(parent_path, tar_file_name)
        with tarfile.open(tar_file_path, "w") as tar:
            for file in self.files:
                tar.add(file.path)

        self.name = tar_file_name
        self.files = None
        self.url = None
        self.path = tar_file_path
        self.is_archive = True

    def unzip(self, extract_directory: Optional[str] = None):
        """
        Extracts a tar archive Directory.
        The extracted files would be saved in the parent directory of
        `self`, unless `extract_directory` argument is specified

        :param extract_directory: the local path to create
            folder Directory at (default = None)
        """
        files = []
        if extract_directory is None:
            extract_directory = os.path.dirname(self.path)

        if not self.is_archive:
            raise ValueError(
                "Attempting to extract tar archive, "
                "but the Directory object is not tar archive"
            )

        name = ".".join(self.name.split(".")[:-2])
        tar = tarfile.open(self.path, "r")
        path = os.path.join(extract_directory, name)

        for member in tar.getmembers():
            member.name = os.path.basename(member.name)
            tar.extract(member=member, path=path)
            files.append(File(name=member.name, path=os.path.join(path, member.name)))
        tar.close()

        self.name = name
        self.files = files
        self.url = None
        self.path = path
        self.is_archive = False

    def __len__(self):
        return len(self.files)

    def _unpack(self):
        # To unpack the Directory the following criteria need to be fulfilled:
        # 1) The Directory needs to be a tar archive
        # 2) The Directory needs to have a `path` attribute.
        return self.is_archive and self.path
