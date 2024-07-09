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
                prefix is None or file.startswith(prefix)) and (suffix is None or file.endswith(suffix)) and not file.endswith(".gitkeep"):
            files.append(file)
    paths = [os.path.join(root, file) for file in files]
    return files, paths


def get_hash_first_file(dir_path, project_name, file_commits, keyword):
    """
        Finds the first file containing GPT-generated code in the commit history that contains a specified keyword.

        Args:
            dir_path (str): The directory path where the project resides.
            project_name (str): The name of the project.
            file_commits (list[dict]): A list of commit dictionaries, each containing 'hash_code' and 'file_path'.
            keyword (str): The keyword to search for within the file contents.

        Returns:
            tuple: A tuple containing:
                - r_index (int): The reverse index of the first matching commit.
                - r_hash (str): The hash code of the first matching commit.
                - keyword_index (int): The index of the keyword within the file, or -1 if not found.
                - matched_keyword (str): The matched keyword, or an empty string if not found.
    """
    r_index = ''
    r_hash = ''
    for index, commit in enumerate(reversed(file_commits)):
        parent_name = commit.get('hash_code')
        analysis_path = Path(dir_path) / project_name / parent_name
        analysis_file_path = analysis_path / commit.get('file_path')
        if analysis_file_path.exists():
            _, keyword_index, matched_keyword = count_effective_lines(analysis_file_path, keyword)
            if keyword_index != -1:
                r_index = -(index + 1)
                r_hash = parent_name
                return r_index, r_hash, keyword_index, matched_keyword

    return r_index, r_hash, -1, ''

