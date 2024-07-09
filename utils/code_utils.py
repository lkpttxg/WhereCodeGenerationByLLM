from config import config, LANGUAGE
from pathlib import Path
import zipfile
import shutil
from .data_read import load_json
import os
import re
import ast
import difflib
import textwrap


def get_code_language(file_path, extra=None):
    """
    get code file language type based on its file path' suffix.
    For example:
    .py -> Python
    .java -> Java
    :param file_path: The file path of the code file
    :return: The programming language of the code file
    """

    # Dictionary mapping file extensions to programming languages
    extension_to_language = {
        '.py': 'Python',
        '.java': 'Java',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript',
        '.mjs': 'JavaScript',
        '.cjs': 'JavaScript',
        '.vue': 'JavaScript',
        '.c': 'C',
        '.cpp': 'C++',
        '.h': 'C++',
        '.cc': 'C++',
        '.hpp': 'C++',
        '.hh': 'C++',
        '.cs': 'C#',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.go': 'Go',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.ts': 'TypeScript',
        '.mts': 'TypeScript',
        '.tsx': 'TypeScript',
        '.rs': 'Rust',
        '.m': 'Objective-C',
        '.dart': 'Dart',
        '.lua': 'Lua',
        '.groovy': 'Groovy',
        '.scala': 'Scala'
    }

    # Extract the file extension from the file path
    import os
    _, file_extension = os.path.splitext(file_path)
    if file_extension == ".h" and extra == "c":
        return "C"
    # Return the corresponding programming language, or 'Unknown' if not found
    return extension_to_language.get(file_extension, 'Unknown')


def get_commit_message(llm_name, language, keyword, project_name, hash_code):
    from .path_utils import get_files_and_path
    _, filepaths = get_files_and_path(config[llm_name][language][keyword]["src"])
    file_path = filepaths[0]
    json_datas = load_json(file_path)
    for index, data in enumerate(json_datas):
        project_name_1 = data.get('project_name')
        if project_name_1 != project_name:
            continue
        file_commits = data.get('file_all_commit_info_list')
        project_commits = data.get('repo_all_commit_info_list')
        for i in range(len(file_commits)):
            hash_code_1 = file_commits[i].get('hash_code')
            if hash_code_1 != hash_code:
                continue
            for j, p_commit in enumerate(project_commits):
                if hash_code == p_commit.get('hash_code'):
                    commit_message = p_commit.get('description')
                    return commit_message

    return ''


def is_commit_message_bug_or_vulnerability(commit_message):
    bug_keywords = [
        'error', 'bug', 'fix', 'issue', 'mistake',
        'incorrect', 'fault', 'defect', 'flaw',
        'vulnerability', 'security', 'exploit', 'patch'
    ]

    # Convert commit_message to lower case to make the search case-insensitive
    commit_message_lower = commit_message.lower()

    # Check if any of the keywords are in the commit message
    for keyword in bug_keywords:
        if keyword in commit_message_lower:
            return True  # A keyword was found, likely a bug fix or vulnerability

    # No keywords found, likely not related to bug fix or vulnerability
    return False


def count_effective_lines(file_path, keyword):
    """
    Count the number of valid lines of code (ignoring empty lines and pure comment lines)
    and find the first occurrence of specific keyword combinations.
    """
    effective_lines = 0
    comment_index = -1
    keyword_pattern = re.compile(
        rf'\b{re.escape(keyword)}\b (?:by|through|using|via|with) (?:chatgpt|copilot|gpt3|gpt4|gpt-3|gpt-4)',
        re.IGNORECASE
    )
    first_match = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        print(f"Unable to read file: {file_path}")
        return effective_lines, comment_index, first_match

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith('#'):
            effective_lines += 1

        if first_match is None:
            match = keyword_pattern.search(line.lower())

            if match:
                comment_index = i + 1
                first_match = match.group()

    return effective_lines, comment_index, first_match


def unzip_and_restructure(zip_filepath, target_path):
    # Create a temporary directory to decompress
    temp_dir = os.path.join(target_path, 'temp_unzip')
    os.makedirs(temp_dir, exist_ok=True)

    # Unzip the files to a temporary directory
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Traverse the unique subdirectory of the temporary directory (assuming only one root directory)
    root_dir = os.listdir(temp_dir)
    if len(root_dir) != 1:
        raise ValueError("The compressed package contains more than one root directory.")

    root_dir_path = os.path.join(temp_dir, root_dir[0])

    # Move the contents of the root directory to the target path
    for item in os.listdir(root_dir_path):
        s = os.path.join(root_dir_path, item)
        d = os.path.join(target_path, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.move(s, d)
        else:
            shutil.move(s, d)

    # Deleting a temporary directory
    shutil.rmtree(temp_dir)


def convert_technical_debt(minutes):
    # Constants
    minutes_per_hour = 60
    hours_per_day = 8
    minutes_per_day = hours_per_day * minutes_per_hour

    # Calculate days and remaining hours
    days = minutes // minutes_per_day
    remaining_minutes = minutes % minutes_per_day
    hours = remaining_minutes // minutes_per_hour
    remaining_minutes = remaining_minutes % minutes_per_hour

    return days, hours, remaining_minutes


def find_metric_value(json_array, target_metric):
    """
    Find the value and period of a specific metric in a JSON array.

    :param json_array: List of JSON objects
    :param target_metric: The metric to search for
    :return: A dictionary with the metric value and period if found, otherwise None
    """
    for obj in json_array:
        if obj.get('metric') == target_metric:
            result = {
                'value': obj.get('value'),
                'period': obj.get('period')
            }
            return result
    return {}


def parse_str_to_arr(arr_str):
    try:
        start_end_line_data = ast.literal_eval(arr_str)
        print("start_end_line_data:", start_end_line_data)

        # Process the data according to the type of the parsing result
        if isinstance(start_end_line_data, list):
            if all(isinstance(i, int) for i in start_end_line_data):
                if len(start_end_line_data) == 2:
                    return [start_end_line_data]
                else:
                    print("Error: One-dimensional array does not have exactly two elements.")
                    return []
            elif all(isinstance(i, list) and len(i) == 2 for i in start_end_line_data):
                return start_end_line_data
            else:
                print("Error: Invalid input format.")
                return []
        else:
            return []
    except (ValueError, SyntaxError):
        print("Error parsing start_end_line_data: {e}")
        return []


def extract_code_and_save(llm_name, language, keyword, file_path, relative_path, start_end_line_list, code_type, is_save_code=False, is_first=True):
    """
        Extract code blocks based on specified line ranges and save them to new files if required.

        Args:
            llm_name (str): The name of the language model.
            language (str): The programming language of the code.
            keyword (str): A keyword used to locate the save directory in the config.
            file_path (str): Path to the file from which code is to be extracted.
            relative_path (str): Relative path used to determine the save directory.
            start_end_line_list (list): List of tuples containing start and end line numbers for each block.
            code_type (str): Type of code ('method' or 'statement') to determine wrapping logic.
            is_save_code (bool): Flag indicating whether to save the extracted code to new files.
            is_first (bool): Flag indicating which directory ('src' or 'res') to use for saving.

        Returns:
            tuple: A tuple containing:
                - extracted_blocks (list): List of extracted code blocks as strings.
                - save_file_paths (list): List of paths to the saved files, if any.
    """
    save_file_paths = []
    try:
        if is_first:
            save_dir = Path(config[llm_name][language][keyword]['code']['src']) / os.path.dirname(relative_path)
        else:
            save_dir = Path(config[llm_name][language][keyword]['code']['res']) / os.path.dirname(relative_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = Path(file_path)
        with file_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        if lines and lines[-1].endswith('\n'):
            lines.append('')
        extracted_blocks = []

        for start_end in start_end_line_list:
            if isinstance(start_end, list) and len(start_end) == 2:
                start, end = start_end
                if 1 <= start <= end <= len(lines):
                    code_block = ''.join(lines[start - 1:end])
                    extracted_blocks.append(code_block)

                    if code_type == 'method' and language == LANGUAGE.Java.value:
                        code_block = f"class A {{\n{code_block}\n}}"
                    if code_type == 'statement' and language == LANGUAGE.Java.value:
                        code_block = f"""
                        class A {{
                            public static void main(String[] args) {{
                                {code_block}
                            }}
                        }}
                        """
                    dedented_code_block = textwrap.dedent(code_block)
                    # Construct the new file name
                    new_file_name = f"{file_path.stem}_[{start}_{end}]{file_path.suffix}"
                    new_file_path = save_dir / new_file_name

                    # Save the code block to the new file
                    if is_save_code:
                        with new_file_path.open('w', encoding='utf-8') as new_file:
                            new_file.write(dedented_code_block)

                    save_file_paths.append(new_file_path)

        return extracted_blocks, save_file_paths

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def examine_matched_code(extracted_code, file_commit_path, start_end_line_list, is_file=False):
    """
        Examine the matched code segments against the specified file content and determine changes.

        Args:
            extracted_code (list): List of code blocks extracted from the source.
            file_commit_path (str): Path to the file to be compared.
            start_end_line_list (list): List of tuples containing start and end line numbers for each block.
            is_file (bool): Flag indicating if the entire file should be compared as a single segment.

        Returns:
            tuple: A tuple containing:
                - found_change (bool): Whether any change was found.
                - blocks_info (list): Information on the best match positions and highest ratios for each block.
    """
    with open(file_commit_path, 'r', encoding='utf-8') as file:
        new_file_content = file.readlines()
    if new_file_content and new_file_content[-1].endswith('\n'):
        new_file_content.append('')

    extracted_code_str = [''.join(block) for block in extracted_code]

    blocks_info = []

    found_change = False

    for block, start_end_line in zip(extracted_code_str, start_end_line_list):
        highest_ratio = 0.0
        best_match_position = (-1, -1)

        block_len = start_end_line[1] - start_end_line[0] + 1
        if is_file:
            segment = ''.join(new_file_content)
            matcher = difflib.SequenceMatcher(None, block, segment)
            ratio = matcher.ratio()

            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match_position = (1, len(new_file_content))

        else:
            for i in range(len(new_file_content) - block_len + 1):
                segment = ''.join(new_file_content[i:i + block_len])
                matcher = difflib.SequenceMatcher(None, block, segment)
                ratio = matcher.ratio()

                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match_position = (i+1, i + block_len)

        blocks_info.append({
            'highest_ratio': highest_ratio,
            'best_match_position': best_match_position
        })

        if highest_ratio < 1.0:
            found_change = True

    return found_change, blocks_info


