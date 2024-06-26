import os
from pathlib import Path
from .code_utils import count_effective_lines


def get_files_and_path(root, filters=None, prefix=None, suffix=None):
    """
    Retrieve the files and their corresponding file paths under the root directory
    :param root: path
    :param filters: Filter the file types, where None indicates no filtering
    :param prefix: prefix
    :param suffix: suffix
    :return: (file_names, file_path)
    """
    files_ = os.listdir(root)
    files = []
    for file in files_:
        if os.path.isfile(os.path.join(root, file)) and (filters is None or filters in file.split(".")[-2]) and (
                prefix is None or file.startswith(prefix)) and (suffix is None or file.endswith(suffix)):
            files.append(file)
    paths = [os.path.join(root, file) for file in files]
    return files, paths


def get_hash_first_file(dir_path, project_name, file_commits, keyword):
    r_index = ''
    r_hash = ''
    for index, commit in enumerate(reversed(file_commits)):
        parent_name = commit.get('hash_code')
        analysis_path = Path(dir_path) / project_name / parent_name
        analysis_file_path = analysis_path / commit.get('file_path')
        if analysis_file_path.exists():
            loc_of_first_commit, keyword_index, matched_keyword = count_effective_lines(analysis_file_path, keyword)
            if keyword_index != -1:
                r_index = -(index + 1)
                r_hash = parent_name
                return r_index, r_hash, keyword_index, matched_keyword

    return r_index, r_hash, -1, ''

