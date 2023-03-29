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

"""
Helpers for building markdown files
"""


from typing import List


__all__ = ["create_markdown_table"]


def create_markdown_table(
    headers: List[str],
    rows: List[List[str]],
) -> str:
    """
    :param headers: table headers
    :param rows: table rows. each row must have the same number of entries as headers
    :return: constructed markdown table string from the headers and rows
    """
    num_columns = len(headers)
    # get target width for each column, minimum base is 3
    column_widths = [max(3, len(header)) for header in headers]
    for idx, row in enumerate(rows):
        if len(row) != num_columns:
            raise ValueError(
                f"Row: {idx} has invalid number of entries {len(row)}. "
                f"Must match number of headers: {num_columns}"
            )
        column_widths = [
            max(len(entry), width) for entry, width in zip(row, column_widths)
        ]

    # add headers to final table
    table_rows = [_create_markdown_table_row(headers, column_widths)]

    # add separator line to final table
    separators = ["-" * width for width in column_widths]
    table_rows.append(_create_markdown_table_row(separators, column_widths))

    # add base rows to final table
    for row in rows:
        table_rows.append(_create_markdown_table_row(row, column_widths))

    return "\n".join(table_rows)


def _create_markdown_table_row(row: List[str], column_widths: List[int]):
    def _pad_entry(item: str, target_width: int) -> str:
        pad_left = target_width - len(item) + 1
        return f" {item}{' ' * pad_left}"

    row_padded = [_pad_entry(entry, width) for entry, width in zip(row, column_widths)]

    # join with surrounding '|'s
    return f"|{'|'.join(row_padded)}|"
