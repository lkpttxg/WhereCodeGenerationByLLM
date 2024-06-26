'''
读取json文件
'''
import ast
import json
import os
import re
import warnings

import numpy as np

from analyze.draw_violinplot import draw_violinplot, draw_violinplot_from_different_metrics, \
    draw_violinplot_manual_vs_gpt
from utils import *
from config import *
import pandas as pd
import utils
from pathlib import Path
import subprocess
import requests
from requests.auth import HTTPBasicAuth
import time
from datetime import datetime
import shutil

warnings.filterwarnings("ignore")

# wherecode: 4d3710f1a2f733d06132ca37de4340fe9b98f0f2
# wherecode: sqa_451bb81b037608d89ab2b31df3d08cee50f16c89

# url: sonar-scanner.bat -D"sonar.projectKey=wherecode" -D"sonar.sources=." -D"sonar.host.url=http://localhost:9000" -D"sonar.login=4d3710f1a2f733d06132ca37de4340fe9b98f0f2" -D"sonar.inclusions=**/*.py"
# url: sonar-scanner.bat -D"sonar.projectKey=wherecode" -D"sonar.sources=./analyze" -D"sonar.host.url=http://localhost:9000" -D"sonar.token=sqa_451bb81b037608d89ab2b31df3d08cee50f16c89" -D"sonar.inclusions=**/*.py"
# Define the command with the necessary variables
# request_files = "your_files_here"  # Replace with your actual files or file pattern
host_url = "http://localhost:9000"
login_token = "4d3710f1a2f733d06132ca37de4340fe9b98f0f2"
inclusions_json = {
    LANGUAGE.Python.value: "**/*.py",
    LANGUAGE.Java.value: "**/*.java",
    LANGUAGE.C.value: "**/*.c,**/*.h,**/*.cc,**/*.hpp,**/*.hh,**/*.cpp",
    LANGUAGE.CPP.value: "**/*.cpp,**/*.h,**/*.cc,**/*.hpp,**/*.hh,**/*.c",
    LANGUAGE.JavaScript.value: "**/*.js,**/*.mjs,**/*.jsx,**/*.cjs,**/*.vue",
    LANGUAGE.CSharp.value: "**/*.cs",
    LANGUAGE.TypeScript.value: "**/*.ts,**/*.mts,**/*.tsx,**/*.css,**/*.html",
}
analyze_lang = {
    LANGUAGE.Python.value: "py",
    LANGUAGE.Java.value: "java",
    LANGUAGE.C.value: "c",
    LANGUAGE.CPP.value: "cpp",
    LANGUAGE.JavaScript.value: "js",
    LANGUAGE.CSharp.value: "cs",
    LANGUAGE.TypeScript.value: "ts",
}
inclusions = "**/*.py"

code_columns_name = ["code_id", 'review_id', 'matched_keyword', 'keyword_index', "project_name", 'hashs',
                     'path_of_first_file_commit', 'code_type', "code_language", 'code_granularity',
                     'code_complexity', 'number_of_file_commit', 'number_of_all_commit', "loc_of_file_commit",
                     "change_loc_of_file_commit",
                     'size_of_file_commit', 'change_size_of_file_commit', 'complexity_of_file_commit',
                     'change_complexity_of_file_commit', 'number_of_bug_or_vulnerability_file_commit',
                     'number_of_bug_or_vulnerability_all_commit',
                     'percentage_of_bug_or_vulnerability_file_commit', 'percentage_of_bug_or_vulnerability_all_commit',
                     'hash_bug_or_vulnerability_file_commit', "change_loc_of_bug_or_vulnerability_file_commit",
                     "type_of_bug_or_vulnerability_file_commit",
                     'be_cloned']

metrics_columns_name = ["code_id", 'review_id', 'matched_keyword', "project_name", 'hash_of_first_file_commit',
                        'keyword_index', "code_language", 'project_lines', 'project_locs', 'project_statements',
                        'project_functions',
                        'project_classes', 'project_files', 'project_density_comments', 'project_comments',
                        'project_duplicated_lines', 'project_duplicated_blocks', 'project_duplicated_files',
                        'project_duplicated_lines_density', 'project_code_smells', 'project_sqale_index',
                        'project_sqale_debt_ratio', 'project_complexity_all', 'project_cognitive_complexity_all',
                        'project_complexity_mean_method', 'project_cognitive_complexity_mean_method', 'code_lines',
                        'code_locs', 'code_statements', 'code_functions',
                        'code_classes', 'code_files', 'code_density_comments', 'code_comments', 'code_duplicated_lines',
                        'code_duplicated_blocks', 'code_duplicated_files',
                        'code_duplicated_lines_density', 'code_code_smells', 'code_sqale_index',
                        'code_sqale_debt_ratio', 'code_complexity_all', 'code_cognitive_complexity_all',
                        'code_complexity_mean_method', 'code_cognitive_complexity_mean_method']


def write_code_analysis_result_to_csv(llm_name, language, keyword):
    filenames, filepaths = get_files_and_path(config[llm_name][language][keyword]["src"])
    filename = filenames[0]
    file_path = filepaths[0]
    print(f'Loading json file-{filename}: {file_path}')
    json_datas = load_json(file_path)

    df = pd.DataFrame(columns=code_columns_name)
    for index, data in enumerate(json_datas):
        project_commits = data.get('repo_all_commit_info_list')
        file_commits = data.get('file_all_commit_info_list')

        # 1. project name
        project_name = data.get('project_name')
        df.loc[index, 'project_name'] = project_name

        # hash_of_first_file_commit (code generated by gpt)
        f_index, f_hash, keyword_index, matched_keyword = get_hash_first_file(
            config[llm_name][language][keyword]["projects"]["src"],
            project_name, file_commits, keyword)
        print("=====Ensure that the first file of the code generated by GPT====")
        print(f"{f_index}-th in file_commits, hash_code: {f_hash}, keyword_index: {keyword_index}")
        df.loc[index, 'keyword_index'] = keyword_index

        if f_index != '':
            if f_index == -1:
                file_commits = file_commits[:]
            else:
                file_commits = file_commits[:f_index + 1]

        hashs = []
        for file_commit in file_commits:
            hashs.append(file_commit.get('hash_code'))

        df.loc[index, 'hashs'] = hashs

        # 2. code language, path_of_first_file_commit
        if len(file_commits) > 0:
            language_type = utils.get_code_language(file_commits[-1].get('file_path'))
            hash_code = file_commits[-1].get('hash_code')
            relative_path = file_commits[-1].get('file_path')
            path_of_first_file_commit = Path(project_name) / hash_code / relative_path
            size_of_first_commit = file_commits[-1].get('size')

            file_download_path = Path(config[llm_name][language][keyword]["projects"]["src"]) / project_name / \
                                 file_commits[-1].get(
                                     'hash_code') / file_commits[-1].get('file_path')

            if file_download_path.exists():
                loc_of_first_commit, keyword_index, matched_keyword = count_effective_lines(file_download_path, keyword)
            else:
                loc_of_first_commit = 0
                keyword_index = -1
                matched_keyword = ''
        else:
            language_type = 'No File'
            path_of_first_file_commit = ""
            size_of_first_commit = 0
            loc_of_first_commit = 0
            keyword_index = -1
            matched_keyword = ''

        df.loc[index, 'code_language'] = language_type
        df.loc[index, 'path_of_first_file_commit'] = path_of_first_file_commit
        df.loc[index, 'keyword_index'] = keyword_index
        df.loc[index, 'matched_keyword'] = matched_keyword

        # 3. size_of_file_commit, change_size_of_file_commit
        size_of_file_commit = []
        change_size_of_file_commit = []
        # 10. loc_of_file_commit, change_loc_of_file_commit
        loc_of_file_commit = []
        change_loc_of_file_commit = []
        # 7. number_of_bug_or_vulnerability_file_commit, path_bug_or_vulnerability_file_commit
        number_of_bug_or_vulnerability_file_commit = 0
        hash_bug_or_vulnerability_file_commit = []
        for i, commit in enumerate(file_commits):
            size_of_file_commit.append(commit.get('size'))
            change_size_of_file_commit.append(commit.get('size') - size_of_first_commit)
            file_download_path = Path(
                config[llm_name][language][keyword]["projects"]["src"]) / project_name / commit.get(
                'hash_code') / commit.get('file_path')
            if file_download_path.exists():
                effective_loc = count_effective_lines(file_download_path, keyword)[0]
                loc_of_file_commit.append(effective_loc)
                change_loc_of_file_commit.append(effective_loc - loc_of_first_commit)
            else:
                loc_of_file_commit.append(0)
                change_loc_of_file_commit.append(0)
            commit_message = ""
            for j, p_commit in enumerate(project_commits):
                if commit.get('hash_code') == p_commit.get('hash_code'):
                    commit_message = p_commit.get('description')
                    break
            if is_commit_message_bug_or_vulnerability(commit_message):
                number_of_bug_or_vulnerability_file_commit += 1
                hash_bug_or_vulnerability_file_commit.append(commit.get('hash_code'))

        df.loc[index, 'size_of_file_commit'] = size_of_file_commit
        df.loc[index, 'change_size_of_file_commit'] = change_size_of_file_commit
        df.loc[index, 'loc_of_file_commit'] = loc_of_file_commit
        df.loc[index, 'change_loc_of_file_commit'] = change_loc_of_file_commit
        df.loc[index, 'number_of_bug_or_vulnerability_file_commit'] = number_of_bug_or_vulnerability_file_commit
        df.loc[index, 'hash_bug_or_vulnerability_file_commit'] = hash_bug_or_vulnerability_file_commit

        # 4. number_of_file_commit
        number_of_file_commit = len(file_commits)
        df.loc[index, 'number_of_file_commit'] = number_of_file_commit
        # 5. number_of_all_commit
        number_of_all_commit = len(project_commits)
        df.loc[index, 'number_of_all_commit'] = number_of_all_commit

        # 6. number_of_bug_or_vulnerability_all_commit
        number_of_bug_or_vulnerability_all_commit = 0
        for i, commit in enumerate(project_commits):
            commit_message = commit.get('description')
            if is_commit_message_bug_or_vulnerability(commit_message):
                number_of_bug_or_vulnerability_all_commit += 1
        df.loc[index, 'number_of_bug_or_vulnerability_all_commit'] = number_of_bug_or_vulnerability_all_commit

        # 8. percentage_of_bug_or_vulnerability_file_commit
        if number_of_file_commit != 0:
            df.loc[
                index, 'percentage_of_bug_or_vulnerability_file_commit'] = number_of_bug_or_vulnerability_file_commit * 1.0 / number_of_file_commit

        # 9. percentage_of_bug_or_vulnerability_all_commit
        if number_of_all_commit != 0:
            df.loc[
                index, 'percentage_of_bug_or_vulnerability_all_commit'] = number_of_bug_or_vulnerability_all_commit * 1.0 / number_of_all_commit

    # get csv file path
    csv_file_name = filename.replace('.json', '.csv')
    csv_file_path = os.path.join(config[llm_name][language][keyword]["res"], csv_file_name)
    # write to csv
    df.to_csv(csv_file_path, index=False)
    print("===========Save File: csv================")
    print("Analysis results have been saved to:", csv_file_path)


def request_sonarqube(llm_name, language, keyword, is_project=False, start=None, end=None):
    filenames, filepaths = get_files_and_path(config[llm_name][language][keyword]["src"])
    filename = filenames[0]
    file_path = filepaths[0]
    print(f'Loading json file-{filename}: {file_path}')
    json_datas = load_json(file_path)

    df = pd.DataFrame(columns=metrics_columns_name)
    for index, data in enumerate(json_datas):
        if start is None:
            start = 0
        if end is None:
            end = len(json_datas)

        if index < start or index >= end:
            continue
        # get project_path
        # 1. project name
        project_name = data.get('project_name')
        df.loc[index, 'project_name'] = project_name

        file_commits = data.get('file_all_commit_info_list')

        # hash_of_first_file_commit (code generated by gpt)
        f_index, f_hash, keyword_index, matched_keyword = get_hash_first_file(
            config[llm_name][language][keyword]["projects"]["src"], project_name, file_commits, keyword)
        print("=====Ensure that the first file of the code generated by GPT====")
        print(f"{f_index}-th in file_commits, hash_code: {f_hash}, keyword_index: {keyword_index}")
        df.loc[index, 'hash_of_first_file_commit'] = f_hash
        df.loc[index, 'keyword_index'] = keyword_index
        df.loc[index, 'matched_keyword'] = matched_keyword

        # if len(file_commits) > 0:
        for index1, commit in enumerate(reversed(file_commits)):
            if commit.get('hash_code') != f_hash:
                continue
            language_type = utils.get_code_language(commit.get('file_path'))
            df.loc[index, 'code_language'] = language_type
            parent_name = commit.get('hash_code')
            # path_of_first_file_commit = Path(project_name) / parent_name / commit.get('file_path')
            # df.loc[index, 'path_of_first_file_commit'] = path_of_first_file_commit
            analysis_path = Path(config[llm_name][language][keyword]["projects"]["src"]) / project_name / parent_name
            analysis_file_path = analysis_path / commit.get('file_path')
            if not analysis_file_path.exists():
                print(f'File {commit.get("file_path")} does not exist in {analysis_path}')
                print(f'This project {index}-th [{project_name}] does not exist')
                continue
            relative_path = '/' + commit.get('file_path')
            os.chdir(analysis_path)

            print(f'--------Analyze project of {index}-th [{project_name}/{parent_name}]: {analysis_path}---------')
            # Construct the analysis project command
            command = [
                "sonar-scanner.bat",
                # f'-Dsonar.projectKey={project_key}',
                f'-Dsonar.sources=.',
                f'-Dsonar.host.url={host_url}',
                f'-Dsonar.login={login_token}',
                f'-Dsonar.inclusions={inclusions}'
            ]

            # Execute the command
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8')
                print("Command executed successfully.")
                # print("Output:", result.stdout)
                # print("Error:", result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
                # print("Output:", e.stdout)
                print("Error:", e.stderr)
                continue

            # Add delay to ensure SonarQube updates
            time.sleep(10)

            if result.returncode == 0:
                data_metrics = api_metrics("")
                if data_metrics != None:
                    print(
                        f'--------Analyze project metrics of {index}-th [{project_name}/{parent_name}]: {analysis_path}----------')
                    # project_lines
                    project_lines = find_metric_value(data_metrics, 'lines').get('value')
                    df.loc[index, 'project_lines'] = project_lines
                    print(f'project_lines: {project_lines}')
                    # project_locs
                    project_locs = find_metric_value(data_metrics, 'ncloc').get('value')
                    df.loc[index, 'project_locs'] = project_locs
                    print(f'project_locs: {project_locs}')
                    # project_statements
                    project_statements = find_metric_value(data_metrics, 'statements').get('value')
                    df.loc[index, 'project_statements'] = project_statements
                    print(f'project_statements: {project_statements}')
                    # project_functions
                    project_functions = find_metric_value(data_metrics, 'functions').get('value')
                    df.loc[index, 'project_functions'] = project_functions
                    print(f'project_functions: {project_functions}')
                    # project_classes
                    project_classes = find_metric_value(data_metrics, 'classes').get('value')
                    df.loc[index, 'project_classes'] = project_classes
                    print(f'project_classes: {project_classes}')
                    # project_files
                    project_files = find_metric_value(data_metrics, 'files').get('value')
                    df.loc[index, 'project_files'] = project_files
                    print(f'project_files: {project_files}')
                    # project_density_comments
                    project_density_comments = find_metric_value(data_metrics, 'comment_lines_density').get('value')
                    df.loc[index, 'project_density_comments'] = project_density_comments
                    print(f'project_density_comments: {project_density_comments}')
                    # project_comments
                    project_comments = find_metric_value(data_metrics, 'comment_lines').get('value')
                    df.loc[index, 'project_comments'] = project_comments
                    print(f'project_comments: {project_comments}')
                    # project_duplicated_lines
                    project_duplicated_lines = find_metric_value(data_metrics, 'duplicated_lines').get('value')
                    df.loc[index, 'project_duplicated_lines'] = project_duplicated_lines
                    print(f'project_duplicated_lines: {project_duplicated_lines}')
                    # project_duplicated_blocks
                    project_duplicated_blocks = find_metric_value(data_metrics, 'duplicated_blocks').get('value')
                    df.loc[index, 'project_duplicated_blocks'] = project_duplicated_blocks
                    print(f'project_duplicated_blocks: {project_duplicated_blocks}')
                    # project_duplicated_files
                    project_duplicated_files = find_metric_value(data_metrics, 'duplicated_files').get('value')
                    df.loc[index, 'project_duplicated_files'] = project_duplicated_files
                    print(f'project_duplicated_files: {project_duplicated_files}')
                    # project_duplicated_lines_density
                    project_duplicated_lines_density = find_metric_value(data_metrics, 'duplicated_lines_density').get(
                        'value')
                    df.loc[index, 'project_duplicated_lines_density'] = project_duplicated_lines_density
                    print(f'project_duplicated_lines_density: {project_duplicated_lines_density}')
                    # project_code_smells
                    project_code_smells = find_metric_value(data_metrics, 'code_smells').get('value')
                    df.loc[index, 'project_code_smells'] = project_code_smells
                    print(f'project_code_smells: {project_code_smells}')
                    # project_sqale_index
                    project_sqale_index = find_metric_value(data_metrics, 'sqale_index').get('value')
                    df.loc[index, 'project_sqale_index'] = project_sqale_index
                    print(f'project_sqale_index: {project_sqale_index}')
                    # project_sqale_debt_ratio
                    project_sqale_debt_ratio = find_metric_value(data_metrics, 'sqale_debt_ratio').get('value')
                    df.loc[index, 'project_sqale_debt_ratio'] = project_sqale_debt_ratio
                    print(f'project_sqale_debt_ratio: {project_sqale_debt_ratio}')
                    # project_complexity_all
                    project_complexity_all = find_metric_value(data_metrics, 'complexity').get('value')
                    df.loc[index, 'project_complexity_all'] = project_complexity_all
                    print(f'project_complexity_all: {project_complexity_all}')
                    # project_cognitive_complexity_all
                    project_cognitive_complexity_all = find_metric_value(data_metrics, 'cognitive_complexity').get(
                        'value')
                    df.loc[index, 'project_cognitive_complexity_all'] = project_cognitive_complexity_all
                    print(f'project_cognitive_complexity_all: {project_cognitive_complexity_all}')
                    # project_complexity_mean_method
                    df.loc[index, 'project_complexity_mean_method'] = calculate_complexity_mean(project_complexity_all,
                                                                                                project_functions)
                    print(
                        f'project_complexity_mean_method: {calculate_complexity_mean(project_complexity_all, project_functions)}')
                    # project_cognitive_complexity_mean_method
                    df.loc[index, 'project_cognitive_complexity_mean_method'] = calculate_complexity_mean(
                        project_cognitive_complexity_all, project_functions)
                    print(
                        f'project_cognitive_complexity_mean_method: {calculate_complexity_mean(project_cognitive_complexity_all, project_functions)}')

                    print(
                        f'--------End analyze metrics of {index}-th [{project_name}/{parent_name}]: {analysis_path}----------')

            print(
                f'--------Analyze file_generated_by_llm of {index}-th [{project_name}/{parent_name}]: {analysis_file_path}----------')
            # Construct the analysis project command
            command = [
                "sonar-scanner.bat",
                # f'-Dsonar.projectKey={project_key}',
                f'-Dsonar.sources=.{relative_path}',
                f'-Dsonar.host.url={host_url}',
                f'-Dsonar.login={login_token}',
                f'-Dsonar.inclusions={inclusions}'
            ]

            # Execute the command
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8')
                print("Command executed successfully.")
                # print("Output:", result.stdout)
                # print("Error:", result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
                # print("Output:", e.stdout)
                print("Error:", e.stderr)
                continue

            # Add delay to ensure SonarQube updates
            time.sleep(15)

            if result.returncode == 0:
                data_metrics = api_metrics("")
                if data_metrics != None:
                    print(
                        f'--------Analyze code metrics of {index}-th [{project_name}/{parent_name}]: {analysis_file_path}----------')
                    # code_lines
                    code_lines = find_metric_value(data_metrics, 'lines').get('value')
                    df.loc[index, 'code_lines'] = code_lines
                    print(f'code_lines: {code_lines}')
                    # code_locs
                    code_locs = find_metric_value(data_metrics, 'ncloc').get('value')
                    df.loc[index, 'code_locs'] = code_locs
                    print(f'code_locs: {code_locs}')
                    # code_statements
                    code_statements = find_metric_value(data_metrics, 'statements').get('value')
                    df.loc[index, 'code_statements'] = code_statements
                    print(f'code_statements: {code_statements}')
                    # code_functions
                    code_functions = find_metric_value(data_metrics, 'functions').get('value')
                    df.loc[index, 'code_functions'] = code_functions
                    print(f'code_functions: {code_functions}')
                    # code_classes
                    code_classes = find_metric_value(data_metrics, 'classes').get('value')
                    df.loc[index, 'code_classes'] = code_classes
                    print(f'code_classes: {code_classes}')
                    # code_files
                    code_files = find_metric_value(data_metrics, 'files').get('value')
                    df.loc[index, 'code_files'] = code_files
                    print(f'code_files: {code_files}')
                    # code_density_comments
                    code_density_comments = find_metric_value(data_metrics, 'comment_lines_density').get('value')
                    df.loc[index, 'code_density_comments'] = code_density_comments
                    print(f'code_density_comments: {code_density_comments}')
                    # code_comments
                    code_comments = find_metric_value(data_metrics, 'comment_lines').get('value')
                    df.loc[index, 'code_comments'] = code_comments
                    print(f'code_comments: {code_comments}')
                    # code_duplicated_lines
                    code_duplicated_lines = find_metric_value(data_metrics, 'duplicated_lines').get('value')
                    df.loc[index, 'code_duplicated_lines'] = code_duplicated_lines
                    print(f'code_duplicated_lines: {code_duplicated_lines}')
                    # code_duplicated_blocks
                    code_duplicated_blocks = find_metric_value(data_metrics, 'duplicated_blocks').get('value')
                    df.loc[index, 'code_duplicated_blocks'] = code_duplicated_blocks
                    print(f'code_duplicated_blocks: {code_duplicated_blocks}')
                    # code_duplicated_files
                    code_duplicated_files = find_metric_value(data_metrics, 'duplicated_files').get('value')
                    df.loc[index, 'code_duplicated_files'] = code_duplicated_files
                    print(f'code_duplicated_files: {code_duplicated_files}')
                    # code_duplicated_lines_density
                    code_duplicated_lines_density = find_metric_value(data_metrics, 'duplicated_lines_density').get(
                        'value')
                    df.loc[index, 'code_duplicated_lines_density'] = code_duplicated_lines_density
                    print(f'code_duplicated_lines_density: {code_duplicated_lines_density}')
                    # code_code_smells
                    code_code_smells = find_metric_value(data_metrics, 'code_smells').get('value')
                    df.loc[index, 'code_code_smells'] = code_code_smells
                    print(f'code_code_smells: {code_code_smells}')
                    # code_sqale_index
                    code_sqale_index = find_metric_value(data_metrics, 'sqale_index').get('value')
                    df.loc[index, 'code_sqale_index'] = code_sqale_index
                    print(f'code_sqale_index: {code_sqale_index}')
                    # code_sqale_debt_ratio
                    code_sqale_debt_ratio = find_metric_value(data_metrics, 'sqale_debt_ratio').get('value')
                    df.loc[index, 'code_sqale_debt_ratio'] = code_sqale_debt_ratio
                    print(f'code_sqale_debt_ratio: {code_sqale_debt_ratio}')
                    # code_complexity_all
                    code_complexity_all = find_metric_value(data_metrics, 'complexity').get('value')
                    df.loc[index, 'code_complexity_all'] = code_complexity_all
                    print(f'code_complexity_all: {code_complexity_all}')
                    # code_cognitive_complexity_all
                    code_cognitive_complexity_all = find_metric_value(data_metrics, 'cognitive_complexity').get(
                        'value')
                    df.loc[index, 'code_cognitive_complexity_all'] = code_cognitive_complexity_all
                    print(f'code_cognitive_complexity_all: {code_cognitive_complexity_all}')
                    # code_complexity_mean_method
                    df.loc[index, 'code_complexity_mean_method'] = calculate_complexity_mean(code_complexity_all,
                                                                                             code_functions)
                    print(
                        f'code_complexity_mean_method: {calculate_complexity_mean(code_complexity_all, code_functions)}')
                    # code_cognitive_complexity_mean_method
                    df.loc[index, 'code_cognitive_complexity_mean_method'] = calculate_complexity_mean(
                        code_cognitive_complexity_all,
                        code_functions)
                    print(
                        f'code_cognitive_complexity_mean_method: {calculate_complexity_mean(code_cognitive_complexity_all, code_functions)}')

                    print(
                        f'--------End analyze metrics of {index}-th [{project_name}/{parent_name}]: {analysis_file_path}----------')

                    csv_file_name = filename.replace('.json', '_metrics.csv')
                    csv_file_path = os.path.join(config[llm_name][language][keyword]["res"], csv_file_name)
                    # write to csv
                    df.to_csv(csv_file_path, index=False)
                    print("===========Save File: csv================")
                    print("Analysis results have been saved to:", csv_file_path)

    csv_file_name = filename.replace('.json', '_metrics.csv')
    csv_file_path = os.path.join(config[llm_name][language][keyword]["res"], csv_file_name)
    # write to csv
    df.to_csv(csv_file_path, index=False)
    print("===========Save File: csv================")
    print("Analysis results have been saved to:", csv_file_path)


def api_metrics(project_key):
    # API URL
    url = "http://localhost:9000/api/measures/component"

    # Query parameters
    params = {
        "component": project_key,
        "metricKeys": (
            "sqale_debt_ratio,comment_lines,comment_lines_density,files,classes,functions,statements,"
            "complexity,cognitive_complexity,alert_status,quality_gate_details,bugs,new_bugs,reliability_rating,"
            "new_reliability_rating,vulnerabilities,new_vulnerabilities,security_rating,new_security_rating,"
            "security_hotspots,new_security_hotspots,security_hotspots_reviewed,new_security_hotspots_reviewed,"
            "security_review_rating,new_security_review_rating,code_smells,new_code_smells,sqale_rating,"
            "new_maintainability_rating,sqale_index,new_technical_debt,coverage,new_coverage,lines_to_cover,"
            "new_lines_to_cover,tests,duplicated_lines,duplicated_lines_density,duplicated_blocks,"
            "ncloc,ncloc_language_distribution,projects,lines,new_lines, duplicated_files"
        )
    }

    # Basic Auth credentials
    username = "admin"
    password = "ll0304051X"

    # Send the request
    response = requests.get(url, params=params, auth=HTTPBasicAuth(username, password))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        print("Request metrics of analysis results was successful.")
        return data.get('component').get('measures')
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return None


def api_create_project(project_key, project_name):
    url = "http://localhost:9000/api/projects/create"

    # Basic Auth credentials
    username = "admin"
    password = "ll0304051X"

    params = {
        "name": project_name,
        "project": project_key
    }

    # Send the request
    response = requests.post(url, params=params, auth=HTTPBasicAuth(username, password))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        print(f"Request creat project {data.get('project')} was successful.")
        return True
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return False


def api_delete_project(project_key):
    url = "http://localhost:9000/api/projects/delete"

    # Basic Auth credentials
    username = "admin"
    password = "ll0304051X"

    params = {
        "project": project_key
    }

    # Send the request
    response = requests.post(url, params=params, auth=HTTPBasicAuth(username, password))

    # Check if the request was successful
    if response.status_code == 204:
        # Parse the JSON response
        print(f"Request delete project {project_key} was successful.")
        return True
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return False


def calculate_complexity_mean(complexity_all, n_functions):
    try:
        return round(float(complexity_all) / float(n_functions), 2)
    except ZeroDivisionError:
        return round(float(complexity_all), 2)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


def analyze_loc_change(llm_name, language, keyword, start=None, end=None, is_save_code=False, is_diff=False,
                       is_download_file=False, is_download_commit=False, is_download_project=False, is_sonarqube=False):
    '''
    First, analyze manual.csv, extract code that generated by chatgpt or copilot, and save to the code folder
    Then, ensure other commit has accurate match the extracted code. If not, record the hash of commit
    :param llm_name:
    :param language:
    :param keyword:
    :param start:
    :param end:
    :param is_diff:
    :param is_download_project:
    :param is_sonarqube:
    :return:
    '''
    filenames, filepaths = get_files_and_path(config[llm_name][language][keyword]["src"])
    filename = filenames[0]
    file_path = filepaths[0]

    csv_file_name = filename.replace('.json', '.csv')
    csv_file_path = os.path.join(config[llm_name][language][keyword]["manual"]['res'], csv_file_name)

    print(f'Loading json file-{filename}: {file_path}')
    json_datas = load_json(file_path)

    df = pd.DataFrame(columns=all_columns)
    flag_empty = True

    if os.path.exists(csv_file_path):
        flag_empty = False
        old_df = pd.read_csv(csv_file_path)
        for col in old_df.columns:
            if col in all_columns:
                df[col] = old_df[col]

    for index, data in enumerate(json_datas):
        if start is None:
            start = 0
        if end is None:
            end = len(json_datas)

        if index < start or index >= end:
            continue

        project_name = data.get('project_name')
        project_commits = data.get('repo_all_commit_info_list')
        file_commits = data.get('file_all_commit_info_list')
        time_format = '%Y-%m-%dT%H:%M:%SZ'
        parsed_time = datetime.strptime(data.get('repo_create_date'), time_format)
        threshold_date = datetime(2023, 2, 1)
        if parsed_time < threshold_date:
            continue
        formatted_time = parsed_time.strftime('%Y-%m-%d %H:%M:%S')

        if is_download_file:
            # download all file commit file
            print(
                f"===================Start download {index}/{len(json_datas)} project: {project_name}===========================")
            for i in range(len(file_commits)):
                file_download_url = file_commits[i].get('download_url')
                parent_name = file_commits[i].get('hash_code')
                relative_file_path = file_commits[i].get('file_path')
                download_local_path = Path(
                    config[llm_name][language][keyword]["projects"][
                        "src"]) / project_name / parent_name / os.path.dirname(
                    relative_file_path)
                try:
                    download_local_path.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    print(f"Error creating directory: {e}")
                    continue
                local_file_path = download_local_path / os.path.basename(relative_file_path)
                print(f"[{i}/{len(file_commits)}] Downloading file:", project_name, "from:", file_download_url, "to:",
                      local_file_path)
                # Attempt to download file with retries
                max_attempts = 30
                attempts = 0
                while attempts < max_attempts:
                    try:
                        with requests.get(file_download_url, stream=True) as r:
                            r.raise_for_status()  # Raises a HTTPError if the response status code is 4XX/5XX
                            with open(local_file_path, 'wb', encoding='utf-8') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                        print("Downloaded file:", project_name + "-" + parent_name, "successfully!")
                        break  # Break the loop if download was successful
                    except Exception as e:
                        attempts += 1
                        print(f"Attempt {attempts} failed with error: {e}")
                        if attempts == max_attempts:
                            print(
                                f"Failed to download file after {max_attempts} attempts. Project: {project_name}, file: {relative_file_path}, index: {i}")

        # hash_of_first_file_commit (code generated by gpt)
        f_index, f_hash, keyword_index, matched_keyword = get_hash_first_file(
            config[llm_name][language][keyword]["projects"]["src"], project_name, file_commits, keyword)
        print(f"=====[{index}/{len(json_datas)}] Ensure that the first file of the code generated by GPT====")
        print(
            f"{f_index}-th in file_commits, matched_keyword: {matched_keyword}, hash_code: {f_hash}, keyword_index: {keyword_index}")

        # 【success get file commits】
        if f_index != '':
            path_of_first_file_commit_value = Path(project_name) / f_hash / file_commits[f_index].get(
                'file_path')
            if f_index == -1:
                file_commits = file_commits[:]
            else:
                file_commits = file_commits[:f_index + 1]
        else:
            file_commits = []
            path_of_first_file_commit_value = ''
            continue

        if is_download_commit:
            for i in range(len(file_commits)):
                hash_code = file_commits[i].get('hash_code')
                commit_message = ""
                for j, p_commit in enumerate(project_commits):
                    if hash_code == p_commit.get('hash_code'):
                        commit_message = p_commit.get('description')
                        break

                relative_file_path = file_commits[i].get('file_path')

                code_file_name = os.path.basename(relative_file_path).split('.')[0]
                save_local_path = Path(config[llm_name][language][keyword]["projects"][
                                           "src"]) / project_name / hash_code / os.path.dirname(
                    relative_file_path)
                save_local_path.mkdir(parents=True, exist_ok=True)

                local_file_path = save_local_path / f'{code_file_name}-commit_info.txt'

                print("Saving file commit message:", project_name, "to:", local_file_path)

                with open(local_file_path, 'w', encoding='utf-8') as f:
                    f.write(commit_message)

                print("Saving file commit message:", project_name + "-" + hash_code, "successfully!")

        try:
            manual_df = pd.read_csv(get_files_and_path(config[llm_name][language][keyword]['manual']['src'])[1][0])
        except UnicodeDecodeError as e:
            print(f"Unicode decode error: {e}")
            byte_position = e.start
            context_bytes = read_problematic_byte(
                get_files_and_path(config[llm_name][language][keyword]['manual']['src'])[1][0], byte_position)
            print(f"Problematic byte context around position {byte_position}: {context_bytes}")
            continue
        manual_df['path_of_first_file_commit'] = manual_df['path_of_first_file_commit'].str.replace('\\', '/')
        # get target line
        filtered_row = manual_df[
            manual_df['path_of_first_file_commit'] == str(path_of_first_file_commit_value).replace('\\', '/')]
        now_row = df[df['path_of_first_file_commit'] == str(path_of_first_file_commit_value).replace('\\', '\\')]
        # get target line index
        if now_row.empty:
            if not flag_empty:
                continue
            now_index = index
        else:
            now_index = now_row.index[0]
        # Extract the start_end_line data for this row
        if not filtered_row.empty:
            code_granularity_data = filtered_row['code_granularity'].values[0]
            start_end_line_data = filtered_row['start_end_line'].values[0]
            start_end_line_list = parse_str_to_arr(start_end_line_data)
            test_code_data = filtered_row['test_code'].values[0]
            regular_expression_data = filtered_row['regular_expression'].values[0]
            # Check whether code_granularity_data and start_end_line_data are both NaN
            if pd.isna(code_granularity_data) or pd.isna(
                    start_end_line_data) or start_end_line_list == [] or code_granularity_data not in ['method', 'file',
                                                                                                       'statement',
                                                                                                       'class']:
                print("Both code_granularity_data orstart_end_line_data are NaN, skipping this row.")
                continue

            df.loc[now_index, 'code_granularity_data'] = code_granularity_data
            df.loc[now_index, 'start_end_line_data'] = start_end_line_data
            df.loc[now_index, 'first_loc'] = get_loc_sum(start_end_line_list)
            df.loc[now_index, 'test_code_data'] = test_code_data
            df.loc[now_index, 'regular_expression_data'] = regular_expression_data
        else:
            print("No matching row found.")
            continue

        language_type = 'No File'
        if len(file_commits) > 0:
            language_type = utils.get_code_language(file_commits[-1].get('file_path'), extra=language)

        # 【success filter project】
        df.loc[now_index, 'index'] = index
        df.loc[now_index, 'project_name'] = project_name
        df.loc[now_index, 'create_time'] = formatted_time
        df.loc[now_index, 'project_language'] = data.get('repo_main_langauge')
        df.loc[now_index, 'contributor'] = data.get('contributor_num')
        df.loc[now_index, 'star'] = data.get('repo_star_num')
        df.loc[now_index, 'fork'] = data.get('repo_fork_num')
        df.loc[now_index, 'issues'] = data.get('repo_issues_num')
        df.loc[now_index, 'watch'] = data.get('repo_watch_num')
        df.loc[now_index, 'project_commits'] = len(data.get('repo_all_commit_info_list'))
        df.loc[now_index, 'code_language'] = language_type
        df.loc[now_index, 'keyword_index'] = keyword_index
        df.loc[now_index, 'matched_keyword'] = matched_keyword
        df.loc[now_index, 'path_of_first_file_commit'] = path_of_first_file_commit_value
        df.loc[now_index, 'number_of_commits'] = len(file_commits)

        # get number_of_bug_or_vulnerability_all_commit
        number_of_bug_or_vulnerability_all_commit = 0
        for i, commit in enumerate(project_commits):
            commit_message = commit.get('description')
            if is_commit_message_bug_or_vulnerability(commit_message):
                number_of_bug_or_vulnerability_all_commit += 1
        df.loc[now_index, 'number_of_bug_or_vulnerability_all_commit'] = number_of_bug_or_vulnerability_all_commit

        absolute_path_of_first_file = Path(
            config[llm_name][language][keyword]["projects"]["src"]) / path_of_first_file_commit_value

        if not os.path.exists(absolute_path_of_first_file):
            print(f"The path {absolute_path_of_first_file} does not exist.")
            continue
        else:
            # extract code
            extracted_code, analysis_file_paths = extract_code_and_save(llm_name, language, keyword,
                                                                        absolute_path_of_first_file,
                                                                        path_of_first_file_commit_value,
                                                                        start_end_line_list, code_granularity_data,
                                                                        is_save_code=is_save_code)

        if not extracted_code or not analysis_file_paths:
            print("No code extracted, skipping this row.")
            continue

        # check whether the code is changed
        if is_diff:
            loc_change_index = []
            loc_change_hash = []
            change_blocks_info = []

            for index1, file_commit in enumerate(reversed(file_commits)):
                # file_path
                file_commit_path = Path(
                    config[llm_name][language][keyword]["projects"]["src"]) / project_name / file_commit.get(
                    'hash_code') / file_commit.get('file_path')

                if not os.path.exists(file_commit_path):
                    continue

                found_change, blocks_info = examine_matched_code(extracted_code, file_commit_path, start_end_line_list,
                                                                 code_granularity_data == 'file')
                # if have been changed
                if found_change:
                    loc_change_index.append(-(index1 + 1))
                    loc_change_hash.append(file_commit.get('hash_code'))
                    change_blocks_info.append(blocks_info)
                    print(f"{-(index1 + 1)}-index/{file_commit.get('hash_code')}, Be changed.")

            df.loc[now_index, 'number_of_change_commit_to_first'] = len(loc_change_hash)
            df.at[now_index, 'change_commit_to_first_index'] = loc_change_index
            df.at[now_index, 'change_commit_to_first_hash'] = loc_change_hash
            df.at[now_index, 'change_commit_to_first_blocks'] = change_blocks_info

            number_of_all_change_commit = 0
            all_change_commit_blocks = []
            all_change_commit_hash = []
            all_change_commit_index = []

            number_of_all_change_fix_commit = 0
            all_change_fix_commit_blocks = []
            all_change_fix_commit_index = []
            all_change_fix_commit_hash = []
            final_change_commit_path = ''

            previous_ratio = None
            for index1, block_info, change_index, change_hash in zip(range(len(change_blocks_info)), change_blocks_info,
                                                                     loc_change_index, loc_change_hash):
                current_ratio = block_info[0]['highest_ratio']

                if index1 == 0 or current_ratio != previous_ratio:
                    all_change_commit_blocks.append(block_info)
                    number_of_all_change_commit += 1
                    all_change_commit_hash.append(change_hash)
                    all_change_commit_index.append(change_index)
                    commit_message = get_commit_message(llm_name, language, keyword, project_name, change_hash)
                    if is_commit_message_bug_or_vulnerability(commit_message):
                        number_of_all_change_fix_commit += 1
                        all_change_fix_commit_blocks.append(block_info)
                        all_change_fix_commit_hash.append(change_hash)
                        all_change_fix_commit_index.append(change_index)

                previous_ratio = current_ratio

            if len(all_change_commit_blocks) > 0:
                final_change_commit_path = Path(project_name) / all_change_commit_hash[-1] / file_commits[-1].get(
                    'file_path')

            df.loc[now_index, 'number_of_all_change_commit'] = number_of_all_change_commit
            df.at[now_index, 'all_change_commit_blocks'] = all_change_commit_blocks
            df.at[now_index, 'all_change_commit_hash'] = all_change_commit_hash
            df.at[now_index, 'all_change_commit_index'] = all_change_commit_index

            df.loc[now_index, 'number_of_all_change_fix_commit'] = number_of_all_change_fix_commit
            df.at[now_index, 'all_change_fix_commit_blocks'] = all_change_fix_commit_blocks
            df.at[now_index, 'all_change_fix_commit_hash'] = all_change_fix_commit_hash
            df.at[now_index, 'all_change_fix_commit_index'] = all_change_fix_commit_index

            df.loc[now_index, 'final_change_commit_path'] = final_change_commit_path

        if is_download_project:
            project_download_url = file_commits[-1].get('project_download_url')
            parent_name = file_commits[-1].get('hash_code')
            download_local_path = Path(config[llm_name][language][keyword]["projects"]["src"]) / project_name
            unzip_path = download_local_path / parent_name
            download_local_path.mkdir(parents=True, exist_ok=True)
            unzip_path.mkdir(parents=True, exist_ok=True)

            download_file_path = download_local_path / f"{parent_name}.zip"

            print(f"Downloading {index}-th project[{project_name}]:", project_name, "from:", project_download_url,
                  "to:", download_file_path)

            # Attempt to download file with retries
            max_attempts = 10
            attempts = 0
            while attempts < max_attempts:
                try:
                    r = requests.get(project_download_url, stream=True)
                    with open(download_file_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)

                    unzip_and_restructure(download_file_path, unzip_path)
                    os.remove(download_file_path)
                    print(f"Download {index}-th project[{project_name}]:", project_name + "-" + parent_name,
                          "successfully!")
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed with error: {e}")
                    if attempts == max_attempts:
                        print(
                            f"Failed to download file after {max_attempts} attempts. Project: {project_name}, Download {index}-th project[{project_name}]:",
                            project_name + "-" + parent_name, "failed!")

        if is_sonarqube:
            parent_name = file_commits[-1].get('hash_code')
            analysis_path = Path(config[llm_name][language][keyword]["projects"]["src"]) / project_name / parent_name
            os.chdir(analysis_path)
            print(f'--------Analyze project of {index}-th [{project_name}/{parent_name}]: {analysis_path}---------')

            # Construct the project key
            project_key = f"{index}-{project_name}-project"
            api_create_project(project_key, project_key)

            empty_dir = "empty"
            if not os.path.exists(empty_dir):
                os.makedirs(empty_dir)

            # Construct the analysis project command
            command = [
                "sonar-scanner.bat",
                f'-Dsonar.projectKey={project_key}',
                f'-Dsonar.sources=.',
                f'-Dsonar.host.url={host_url}',
                f'-Dsonar.login={login_token}',
                f'-Dsonar.inclusions={inclusions_json.get(language)}',
                '-Dsonar.java.binaries=empty',
                '-Dsonar.projectVersion=1.0',
            ]

            # Execute the command
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8')
                print("Command executed successfully.")
                # print("Output:", result.stdout)
                # print("Error:", result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
                # print("Output:", e.stdout)
                print("Error:", e.stderr)
                continue
            finally:
                if os.path.exists(empty_dir):
                    shutil.rmtree(empty_dir)

            # Add delay to ensure SonarQube updates
            is_analysis_finished(project_key)

            if result.returncode == 0:
                data_metrics = api_metrics(project_key)
                if data_metrics is not None:
                    print(
                        f'--------Analyze project metrics of {index}-th [{project_name}/{parent_name}]: {analysis_path}----------')
                    # project_lines
                    project_lines = find_metric_value(data_metrics, 'lines').get('value')
                    df.loc[now_index, 'project_lines'] = project_lines
                    print(f'project_lines: {project_lines}')
                    # project_locs
                    project_locs = find_metric_value(data_metrics, 'ncloc').get('value')
                    df.loc[now_index, 'project_locs'] = project_locs
                    print(f'project_locs: {project_locs}')
                    # project_statements
                    project_statements = find_metric_value(data_metrics, 'statements').get('value')
                    df.loc[now_index, 'project_statements'] = project_statements
                    print(f'project_statements: {project_statements}')
                    # project_functions
                    project_functions = find_metric_value(data_metrics, 'functions').get('value')
                    df.loc[now_index, 'project_functions'] = project_functions
                    print(f'project_functions: {project_functions}')
                    # project_classes
                    project_classes = find_metric_value(data_metrics, 'classes').get('value')
                    df.loc[now_index, 'project_classes'] = project_classes
                    print(f'project_classes: {project_classes}')
                    # project_files
                    project_files = find_metric_value(data_metrics, 'files').get('value')
                    df.loc[now_index, 'project_files'] = project_files
                    print(f'project_files: {project_files}')
                    # project_density_comments
                    project_density_comments = find_metric_value(data_metrics, 'comment_lines_density').get('value')
                    df.loc[now_index, 'project_density_comments'] = project_density_comments
                    print(f'project_density_comments: {project_density_comments}')
                    # project_comments
                    project_comments = find_metric_value(data_metrics, 'comment_lines').get('value')
                    df.loc[now_index, 'project_comments'] = project_comments
                    print(f'project_comments: {project_comments}')
                    # project_duplicated_lines
                    project_duplicated_lines = find_metric_value(data_metrics, 'duplicated_lines').get('value')
                    df.loc[now_index, 'project_duplicated_lines'] = project_duplicated_lines
                    print(f'project_duplicated_lines: {project_duplicated_lines}')
                    # project_duplicated_blocks
                    project_duplicated_blocks = find_metric_value(data_metrics, 'duplicated_blocks').get('value')
                    df.loc[now_index, 'project_duplicated_blocks'] = project_duplicated_blocks
                    print(f'project_duplicated_blocks: {project_duplicated_blocks}')
                    # project_duplicated_files
                    project_duplicated_files = find_metric_value(data_metrics, 'duplicated_files').get('value')
                    df.loc[now_index, 'project_duplicated_files'] = project_duplicated_files
                    print(f'project_duplicated_files: {project_duplicated_files}')
                    # project_duplicated_lines_density
                    project_duplicated_lines_density = find_metric_value(data_metrics, 'duplicated_lines_density').get(
                        'value')
                    df.loc[now_index, 'project_duplicated_lines_density'] = project_duplicated_lines_density
                    print(f'project_duplicated_lines_density: {project_duplicated_lines_density}')
                    # project_vulnerability
                    project_vulnerability = find_metric_value(data_metrics, 'vulnerabilities').get('value')
                    df.loc[now_index, 'project_vulnerability'] = project_vulnerability
                    print(f'project_vulnerability: {project_vulnerability}')
                    # project_bugs
                    project_bugs = find_metric_value(data_metrics, 'bugs').get('value')
                    df.loc[now_index, 'project_bugs'] = project_bugs
                    print(f'project_bugs: {project_bugs}')
                    # project_code_smells
                    project_code_smells = find_metric_value(data_metrics, 'code_smells').get('value')
                    df.loc[now_index, 'project_code_smells'] = project_code_smells
                    print(f'project_code_smells: {project_code_smells}')
                    # project_sqale_index
                    project_sqale_index = find_metric_value(data_metrics, 'sqale_index').get('value')
                    df.loc[now_index, 'project_sqale_index'] = project_sqale_index
                    print(f'project_sqale_index: {project_sqale_index}')
                    # project_sqale_debt_ratio
                    project_sqale_debt_ratio = find_metric_value(data_metrics, 'sqale_debt_ratio').get('value')
                    df.loc[now_index, 'project_sqale_debt_ratio'] = project_sqale_debt_ratio
                    print(f'project_sqale_debt_ratio: {project_sqale_debt_ratio}')
                    # project_complexity_all
                    project_complexity_all = find_metric_value(data_metrics, 'complexity').get('value')
                    df.loc[now_index, 'project_complexity_all'] = project_complexity_all
                    print(f'project_complexity_all: {project_complexity_all}')
                    # project_cognitive_complexity_all
                    project_cognitive_complexity_all = find_metric_value(data_metrics, 'cognitive_complexity').get(
                        'value')
                    df.loc[now_index, 'project_cognitive_complexity_all'] = project_cognitive_complexity_all
                    print(f'project_cognitive_complexity_all: {project_cognitive_complexity_all}')
                    # project_complexity_mean_method
                    df.loc[now_index, 'project_complexity_mean_method'] = calculate_complexity_mean(
                        project_complexity_all,
                        project_functions)
                    print(
                        f'project_complexity_mean_method: {calculate_complexity_mean(project_complexity_all, project_functions)}')
                    # project_cognitive_complexity_mean_method
                    df.loc[now_index, 'project_cognitive_complexity_mean_method'] = calculate_complexity_mean(
                        project_cognitive_complexity_all, project_functions)
                    print(
                        f'project_cognitive_complexity_mean_method: {calculate_complexity_mean(project_cognitive_complexity_all, project_functions)}')

                    print(
                        f'--------End analyze metrics of {index}-th [{project_name}/{parent_name}]: {analysis_path}----------')

            # Delete the project if it already exists
            api_delete_project(project_key)

            # Construct the project key
            project_key = f"{index}-{project_name}-code"
            api_create_project(project_key, project_key)

            relative_path = (Path(config[llm_name][language][keyword]["code"]["src"]) / project_name / parent_name /
                             file_commits[-1].get('file_path')).parent
            os.chdir(relative_path)
            print(
                f'--------Start analyze code metrics of {index}-th [{project_name}/{parent_name}]: {relative_path} ----------')
            inclusions_arr = []
            for analysis_file_path in analysis_file_paths:
                inclusions_arr.append(os.path.basename(analysis_file_path))
            code_inclusions = ",".join(inclusions_arr)

            empty_dir = "empty"
            if not os.path.exists(empty_dir):
                os.makedirs(empty_dir)

            # Construct the analysis project command
            command = [
                "sonar-scanner.bat",
                f'-Dsonar.projectKey={project_key}',
                f'-Dsonar.sources=.',
                f'-Dsonar.host.url={host_url}',
                f'-Dsonar.login={login_token}',
                f'-Dsonar.inclusions={code_inclusions}',
                f'-Dsonar.java.binaries={empty_dir}'
            ]

            # Execute the command
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8')
                print("Command executed successfully.")
                # print("Output:", result.stdout)
                # print("Error:", result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
                # print("Output:", e.stdout)
                print("Error:", e.stderr)
                continue
            finally:
                if os.path.exists(empty_dir):
                    shutil.rmtree(empty_dir)

            # Add delay to ensure SonarQube updates
            is_analysis_finished(project_key)

            if result.returncode == 0:
                data_metrics = api_metrics(project_key)
                if data_metrics != None:
                    print(
                        f'--------Analyze code metrics of {index}-th [{project_name}/{parent_name}]: {relative_path}----------')
                    # code_lines
                    code_lines = find_metric_value(data_metrics, 'lines').get('value')
                    if language == LANGUAGE.Java.value and code_granularity_data == 'method':
                        if code_lines is not None and code_lines.strip().isdigit():
                            code_lines = int(code_lines) - 2 * len(analysis_file_paths)
                    if language == LANGUAGE.Java.value and code_granularity_data == 'statement':
                        if code_lines is not None and code_lines.strip().isdigit():
                            code_lines = int(code_lines) - 4 * len(analysis_file_paths)
                    df.loc[now_index, 'code_lines'] = code_lines
                    print(f'code_lines: {code_lines}')
                    # code_locs
                    code_locs = find_metric_value(data_metrics, 'ncloc').get('value')
                    if language == LANGUAGE.Java.value and code_granularity_data == 'method':
                        if code_locs is not None and code_locs.strip().isdigit():
                            code_locs = int(code_locs) - 2 * len(analysis_file_paths)
                    if language == LANGUAGE.Java.value and code_granularity_data == 'statement':
                        if code_locs is not None and code_locs.strip().isdigit():
                            code_locs = int(code_locs) - 4 * len(analysis_file_paths)
                    df.loc[now_index, 'code_locs'] = code_locs
                    print(f'code_locs: {code_locs}')
                    # code_statements
                    code_statements = find_metric_value(data_metrics, 'statements').get('value')

                    df.loc[now_index, 'code_statements'] = code_statements
                    print(f'code_statements: {code_statements}')
                    # code_functions
                    code_functions = find_metric_value(data_metrics, 'functions').get('value')
                    if language == LANGUAGE.Java.value and code_granularity_data == 'statement':
                        if code_functions is not None and code_functions.strip().isdigit():
                            code_functions = int(code_functions) - len(analysis_file_paths)
                    df.loc[now_index, 'code_functions'] = code_functions
                    print(f'code_functions: {code_functions}')
                    # code_classes
                    code_classes = find_metric_value(data_metrics, 'classes').get('value')
                    if language == LANGUAGE.Java.value and (
                            code_granularity_data == 'method' or code_granularity_data == 'statement'):
                        if code_classes is not None and code_classes.strip().isdigit():
                            code_classes = int(code_classes) - len(analysis_file_paths)

                    df.loc[now_index, 'code_classes'] = code_classes
                    print(f'code_classes: {code_classes}')
                    # code_files
                    # code_files = find_metric_value(data_metrics, 'files').get('value')
                    code_files = 1
                    df.loc[now_index, 'code_files'] = code_files
                    print(f'code_files: {code_files}')
                    # code_density_comments
                    code_density_comments = find_metric_value(data_metrics, 'comment_lines_density').get('value')
                    df.loc[now_index, 'code_density_comments'] = code_density_comments
                    print(f'code_density_comments: {code_density_comments}')
                    # code_comments
                    code_comments = find_metric_value(data_metrics, 'comment_lines').get('value')
                    df.loc[now_index, 'code_comments'] = code_comments
                    print(f'code_comments: {code_comments}')
                    # code_duplicated_lines
                    code_duplicated_lines = find_metric_value(data_metrics, 'duplicated_lines').get('value')
                    df.loc[now_index, 'code_duplicated_lines'] = code_duplicated_lines
                    print(f'code_duplicated_lines: {code_duplicated_lines}')
                    # code_duplicated_blocks
                    code_duplicated_blocks = find_metric_value(data_metrics, 'duplicated_blocks').get('value')
                    df.loc[now_index, 'code_duplicated_blocks'] = code_duplicated_blocks
                    print(f'code_duplicated_blocks: {code_duplicated_blocks}')
                    # code_duplicated_files
                    code_duplicated_files = find_metric_value(data_metrics, 'duplicated_files').get('value')
                    df.loc[now_index, 'code_duplicated_files'] = code_duplicated_files
                    print(f'code_duplicated_files: {code_duplicated_files}')
                    # code_duplicated_lines_density
                    code_duplicated_lines_density = find_metric_value(data_metrics, 'duplicated_lines_density').get(
                        'value')
                    df.loc[now_index, 'code_duplicated_lines_density'] = code_duplicated_lines_density
                    print(f'code_duplicated_lines_density: {code_duplicated_lines_density}')
                    # code vulnerability
                    code_vulnerability = find_metric_value(data_metrics, 'vulnerabilities').get('value')
                    df.loc[now_index, 'code_vulnerability'] = code_vulnerability
                    print(f'code_vulnerability: {code_vulnerability}')
                    # code bugs
                    code_bugs = find_metric_value(data_metrics, 'bugs').get('value')
                    df.loc[now_index, 'code_bugs'] = code_bugs
                    print(f'code_bugs: {code_bugs}')
                    # code_code_smells
                    code_code_smells = find_metric_value(data_metrics, 'code_smells').get('value')
                    df.loc[now_index, 'code_code_smells'] = code_code_smells
                    print(f'code_code_smells: {code_code_smells}')
                    # code_sqale_index
                    code_sqale_index = find_metric_value(data_metrics, 'sqale_index').get('value')
                    df.loc[now_index, 'code_sqale_index'] = code_sqale_index
                    print(f'code_sqale_index: {code_sqale_index}')
                    # code_sqale_debt_ratio
                    code_sqale_debt_ratio = find_metric_value(data_metrics, 'sqale_debt_ratio').get('value')
                    df.loc[now_index, 'code_sqale_debt_ratio'] = code_sqale_debt_ratio
                    print(f'code_sqale_debt_ratio: {code_sqale_debt_ratio}')
                    # code_complexity_all
                    code_complexity_all = find_metric_value(data_metrics, 'complexity').get('value')
                    if language == LANGUAGE.Java.value and (code_granularity_data == 'statement'):
                        if code_complexity_all is not None and code_complexity_all.strip().isdigit():
                            code_complexity_all = int(code_complexity_all) - len(analysis_file_paths)
                    df.loc[now_index, 'code_complexity_all'] = code_complexity_all
                    print(f'code_complexity_all: {code_complexity_all}')
                    # code_cognitive_complexity_all
                    code_cognitive_complexity_all = find_metric_value(data_metrics, 'cognitive_complexity').get(
                        'value')
                    df.loc[now_index, 'code_cognitive_complexity_all'] = code_cognitive_complexity_all
                    print(f'code_cognitive_complexity_all: {code_cognitive_complexity_all}')
                    # code_complexity_mean_method
                    df.loc[now_index, 'code_complexity_mean_method'] = calculate_complexity_mean(code_complexity_all,
                                                                                                 code_functions)
                    print(
                        f'code_complexity_mean_method: {calculate_complexity_mean(code_complexity_all, code_functions)}')
                    # code_cognitive_complexity_mean_method
                    df.loc[now_index, 'code_cognitive_complexity_mean_method'] = calculate_complexity_mean(
                        code_cognitive_complexity_all,
                        code_functions)
                    print(
                        f'code_cognitive_complexity_mean_method: {calculate_complexity_mean(code_cognitive_complexity_all, code_functions)}')

                    print(
                        f'--------End analyze metrics of {index}-th [{project_name}/{parent_name}]: {analysis_file_path}----------')

            # Delete the project if it already exists
            api_delete_project(project_key)

        # write to csv
        df.to_csv(csv_file_path, index=False)
        print("===========Save File: csv================")
        print("Analysis results have been saved to:", csv_file_path)

    # write to csv
    df.to_csv(csv_file_path, index=False)
    print("===========Save File: csv================")
    print("Analysis results have been saved to:", csv_file_path)


def is_analysis_finished(project_key):
    url = f"http://localhost:9000/api/ce/component"

    # Basic Auth credentials
    username = "admin"
    password = "ll0304051X"

    params = {
        "component": project_key
    }

    while True:
        response = requests.get(url, params=params, auth=HTTPBasicAuth(username, password))

        if response.status_code == 200:
            data = response.json()
            current_task = data.get('current', None)
            if current_task and current_task['status'] == 'SUCCESS':
                print("Analysis is finished.")
                return True
            elif current_task and current_task['status'] == 'PENDING':
                print("Analysis is still pending. Waiting...")
            elif current_task and current_task['status'] == 'IN_PROGRESS':
                print("Analysis is in progress. Waiting...")
            else:
                print("No current task or unknown status. Waiting...")
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)

        # Wait for a while before checking again
        time.sleep(5)


def get_loc_sum(start_end_line_list):
    total_lines = 0
    for start_end in start_end_line_list:
        start, end = start_end
        total_lines += (end - start + 1)
    return total_lines


def read_problematic_byte(file_path, byte_position, window=20):
    with open(file_path, 'rb') as file:
        # 移动到问题字节位置
        file.seek(byte_position - window // 2)
        # 读取上下文字节
        context = file.read(window)
        return context


def splice_filter_project(llm_name, language):
    res_dir_path = Path(config[llm_name][language]['res'])
    filename = f"{language}_data.csv"

    # List of directories and keyword types
    keyword_dirs = {
        KEYWORD.Authored: Path(config[llm_name][language][KEYWORD.Authored.value]['manual']['res']),
        KEYWORD.Coded: Path(config[llm_name][language][KEYWORD.Coded.value]['manual']['res']),
        KEYWORD.Created: Path(config[llm_name][language][KEYWORD.Created.value]['manual']['res']),
        KEYWORD.Generated: Path(config[llm_name][language][KEYWORD.Generated.value]['manual']['res']),
        KEYWORD.Implemented: Path(config[llm_name][language][KEYWORD.Implemented.value]['manual']['res']),
        KEYWORD.Written: Path(config[llm_name][language][KEYWORD.Written.value]['manual']['res'])
    }

    # Collect paths
    csv_file_paths = []
    for key, dir_path in keyword_dirs.items():
        try:
            csv_file_path = get_files_and_path(dir_path)[1][0]
            filename = get_files_and_path(dir_path)[0][0]
            print(f"Reading csv file: {csv_file_path}")
            csv_file_paths.append(csv_file_path)
        except IndexError:
            print(f"No CSV file found in directory: {dir_path}")

    res_save_path = res_dir_path / filename.replace('.csv', '_all.csv')

    # splice six csv to res_save_path
    # read all csv
    df_list = []
    for csv_file_path in csv_file_paths:
        try:
            df = pd.read_csv(csv_file_path)
            df_list.append(df)
        except FileNotFoundError:
            print(f"File not found: {csv_file_path}")
    if df_list:
        all_data = pd.concat(df_list, ignore_index=True)
        filtered_data = all_data[(all_data['create_time'] >= '2023-02-01') & (all_data['project_locs'].notna()) & (all_data['code_language'] != 'Unknown')]
        filtered_data.to_csv(res_save_path, index=False)
        print("Splice and filter project successfully!")
        print("Saved to:", res_save_path)
    else:
        print("No CSV files to process.")


def statistic_code_function_type(llm_name, language, only_bug_change=False, top_percent=None):
    filenames, filepaths = get_files_and_path(config[llm_name][language]['res'])
    filename = filenames[0]
    filepath = filepaths[0]
    print("Reading csv file:", filepath)
    df = pd.read_csv(filepath)

    if only_bug_change:
        df = df[df["real_fixed_commit_number"] > 0]

    if top_percent:
        remove_dup_project_df = df.drop_duplicates(subset=['project_name'])
        top_percent_threshold = remove_dup_project_df['star'].quantile(1-top_percent)
        print("top_percent_threshold", top_percent_threshold)
        df = df[df['star'] >= top_percent_threshold]

    # df根据 star 列的大小，过滤出前30%的最高star的项目
    # top_30_percent_threshold = df['star'].quantile(0.7)
    # df_top_30 = df[df['star'] >= top_30_percent_threshold]

    # column: code_func_type
    # expanded_types = df_top_30['code_func_type'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
    expanded_types = df['code_func_type'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
    expanded_types.name = 'code_func_type'

    value_counts = expanded_types.value_counts().reset_index()
    value_counts.columns = ['code_func_type', 'count']
    value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum()) * 100

    # 然后我希望，哪几个code_func_type归为一个code_func_type，这几个统计到一起
    merge_dict = {
        '机器学习和深度学习模型代码': ['机器学习和深度学习模型代码', '图像处理和计算机视觉代码', '自然语言处理（NLP）和文本处理代码', '文本处理代码'],
        '业务逻辑代码': ['业务逻辑代码', '加密货币代码', '机器人控制代码', '音乐生成代码'],
        '安全类型的代码': ['安全类型的代码', '内存分配和释放相关的代码'],
        '算法和数据结构实现代码': ['算法和数据结构实现代码', '数据结构和算法实现代码'],
        'Program input code, i.e., variable assignments and regular expressions': ['Program input code, i.e., variable assignments'],
        'Data processing and transformation': ['未知']
    }

    def merge_types(row, column):
        for new_type, old_types in merge_dict.items():
            if row[column] in old_types:
                return new_type
        return row[column]

    value_counts['code_func_type'] = value_counts.apply(lambda row: merge_types(row, 'code_func_type'), axis=1)
    value_counts = value_counts.groupby('code_func_type').agg({'count': 'sum'}).reset_index()
    value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum()) * 100

    output_filepath = Path(config[llm_name][language]['visualization']['src']) / filename.replace(".csv", f"_type_counts_bug_change_{only_bug_change}_percent_{top_percent}.csv")

    value_counts.to_csv(output_filepath, index=False, encoding='utf-8-sig')


    print("=========== Save File: csv ================")
    print("Statistics results have been saved to:", output_filepath)

    # expanded_change_types = df_top_30['final_change_type'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
    expanded_change_types = df['final_change_type'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
    expanded_change_types.name = 'final_change_type'

    # Remove excluded types
    expanded_change_types = expanded_change_types[~expanded_change_types.isin(["Bug修复", 'bug修复'])]

    value_counts = expanded_change_types.value_counts().reset_index()
    value_counts.columns = ['final_change_type', 'count']
    value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum()) * 100

    merge_dict = {
        '优化': ['优化', '性能优化', '用户体验优化'],
        '样式调整': ['样式调整', '无变更', '无修改'],
        '其他': ['依赖更新', '版本控制', '国际化和本地化', '测试']
    }

    value_counts['final_change_type'] = value_counts.apply(lambda row: merge_types(row, 'final_change_type'), axis=1)
    value_counts = value_counts.groupby('final_change_type').agg({'count': 'sum'}).reset_index()
    value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum()) * 100

    output_filepath = Path(config[llm_name][language]['visualization']['src']) / filename.replace(".csv",
                                                                                                  f"_change_type_counts_bug_change_{only_bug_change}_percent_{top_percent}.csv")
    value_counts.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    print("=========== Save File: csv ================")
    print("Statistics results have been saved to:", output_filepath)


def statistic_bug_fix_type(llm_name, language):
    filenames, filepaths = get_files_and_path(config[llm_name][language]['res'])
    filename = filenames[0]
    filepath = filepaths[0]
    print("Reading csv file:", filepath)
    df = pd.read_csv(filepath)

    # df根据 star 列的大小，过滤出前30%的最高star的项目
    # top_30_percent_threshold = df['star'].quantile(0.7)
    # df_top_30 = df[df['star'] >= top_30_percent_threshold]

    # column: real_fixed_commit_reason，一个单元格的内容为[[代码可维护性问题, 接口和依赖管理问题], [接口和依赖管理问题], [代码可维护性问题]]或者[接口和依赖管理问题]。
    # 请先解析这个为数组，其中二维数组说明fixed_commit为多个，一维数组为1个。然后每个fixed_commit可能有多个修复的原因。
    # 请统计总共有多少个fixed_commit,然后再分门别类的统计不同fixed_reason分别有多少个，然后存入csv中
    # Parse 'real_fixed_commit_reason' column
    def parse_fixed_commit_reason(cell):
        if isinstance(cell, str):
            # Replace Chinese commas with English commas
            cell = re.sub(r'，', ',', cell)

            # Add quotes around words that are not numbers or already quoted
            cell = re.sub(r"(?<!['\"])(\b[a-zA-Z_一-龥]+\b)(?!['\"])", r'"\1"', cell)

            try:
                return ast.literal_eval(cell)
            except (ValueError, SyntaxError) as e:
                print("Error parsing cell:", e)
                return None
        return cell

    df['real_fixed_commit_reason'] = df['real_fixed_commit_reason'].apply(parse_fixed_commit_reason)

    # Count total number of fixed commits and categorize fixed reasons
    total_fixed_commits = 0
    fixed_reason_counts = {}

    for reasons in df['real_fixed_commit_reason']:
        if isinstance(reasons, list):
            if all(isinstance(reason, list) for reason in reasons):
                total_fixed_commits += len(reasons)
                for reason_list in reasons:
                    for reason in reason_list:
                        if reason in fixed_reason_counts:
                            fixed_reason_counts[reason] += 1
                        else:
                            fixed_reason_counts[reason] = 1
            else:
                total_fixed_commits += 1
                for reason in reasons:
                    if reason in fixed_reason_counts:
                        fixed_reason_counts[reason] += 1
                    else:
                        fixed_reason_counts[reason] = 1


    print("total_fixed_commits:" + str(total_fixed_commits))
    # Calculate the percentage for each fixed reason
    fixed_reason_percentages = {reason: (count / total_fixed_commits) * 100 for reason, count in
                                fixed_reason_counts.items()}

    # Prepare the results to be saved in a CSV file
    results = {
        'Fixed Reason': list(fixed_reason_counts.keys()),
        'Count': list(fixed_reason_counts.values()),
        'Percentage': [fixed_reason_percentages[reason] for reason in fixed_reason_counts.keys()]
    }
    results_df = pd.DataFrame(results)

    # Save results to CSV
    output_filepath = Path(config[llm_name][language]['visualization']['src']) / filename.replace(".csv",
                                                                                                  "_fixed_reason_statistics.csv")
    results_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    print(f"Results saved to: {output_filepath}")


def calculate_various_measures(llm_name, language, columns=('contributor', 'star', 'fork', 'issues', 'watch',
                                                            'project_commits',
                                                            'number_of_bug_or_vulnerability_all_commit',
                                                            'first_loc', 'number_of_commits', 'project_files',
                                                            'project_lines',
                                                            'project_locs', 'project_statements', 'project_functions',
                                                            'project_code_smells',
                                                            'project_complexity_all',
                                                            'project_cognitive_complexity_all',
                                                            'number_of_all_change_commit', 'real_fixed_commit_number',
                                                            'code_lines', 'code_locs', 'code_statements',
                                                            'code_functions', 'code_code_smells', 'code_complexity_all',
                                                            'code_cognitive_complexity_all', 'number_of_all_change_commit', 'final_loc_add', 'final_loc_minus'), only_change=False, top_percent=None):
    filenames, filepaths = get_files_and_path(config[llm_name][language]['res'])
    filename = filenames[0]
    filepath = filepaths[0]
    print("Reading csv file:", filepath)
    df = pd.read_csv(filepath)

    if only_change:
        df = df[df['number_of_all_change_commit'] > 0]

    if top_percent:
        remove_dup_project_df = df.drop_duplicates(subset=['project_name'])
        top_percent_threshold = remove_dup_project_df['star'].quantile(1 - top_percent)
        print("top_percent_threshold", top_percent_threshold)
        df = df[df['star'] >= top_percent_threshold]

    if columns is None:
        columns = df.columns.tolist()

    measures = {
        'measure': ['median', 'mean', 'max', 'min', 'std'],
    }

    for col in columns:
        if col in df.columns:
            if col in ['number_of_all_change_commit', 'real_fixed_commit_number']:
                clean_data = df[col].dropna().loc[lambda x: x > 0]
            else:
                clean_data = df[col].dropna()
            measures[col] = [
                clean_data.median(),
                clean_data.mean(),
                clean_data.max(),
                clean_data.min(),
                clean_data.std()
            ]

    for index, granularity_name in enumerate(["file", "class", "method", "statement"]):
        filter_df = df[df["code_granularity_data"] == granularity_name]

        filter_loc_df = filter_df["code_locs"].dropna()
        measures[f"granularity_{granularity_name}_loc"] = [
            filter_loc_df.median(),
            filter_loc_df.mean(),
            filter_loc_df.max(),
            filter_loc_df.min(),
            filter_loc_df.std()
        ]

        filter_cc_df = filter_df["code_complexity_all"].dropna()
        measures[f"granularity_{granularity_name}_cc"] = [
            filter_cc_df.median(),
            filter_cc_df.mean(),
            filter_cc_df.max(),
            filter_cc_df.min(),
            filter_cc_df.std()
        ]

        filter_congc_df = filter_df["code_cognitive_complexity_all"].dropna()
        measures[f"granularity_{granularity_name}_congc"] = [
            filter_congc_df.median(),
            filter_congc_df.mean(),
            filter_congc_df.max(),
            filter_congc_df.min(),
            filter_congc_df.std()
        ]

    # clean_data = df['number_of_all_change_commit'].dropna()

    result_df = pd.DataFrame(measures)
    res_dir_path = Path(config[llm_name][language]['visualization']['src'])
    res_save_path = res_dir_path / filename.replace('.csv', f'_measures_change_{only_change}_percent_{top_percent}.csv')
    result_df.to_csv(res_save_path, index=False)
    print(f"Measures saved to {res_save_path}")


def process_final_csv(llm_name, language, extracted_final_change_code=False, cal_complexity=False, set_url=False):
    filenames, filepaths = get_files_and_path(config[llm_name][language]['res'])
    filename = filenames[0]
    filepath = filepaths[0]
    print("Reading csv file:", filepath)
    old_df = pd.read_csv(filepath)
    df = pd.DataFrame(columns=all_columns)

    for col in old_df.columns:
        if col in all_columns:
            df[col] = old_df[col]

    if set_url:
        for index, row in df.iterrows():
            language_ = language
            code_language = row['code_language']
            if language_ == "cpp" and code_language == "C":
                language_ = LANGUAGE.C.value
            matched_keyword = row['matched_keyword']
            key_word = matched_keyword.split(" ")[0] if isinstance(matched_keyword, str) else None

            first_path = row['path_of_first_file_commit']

            # url_of_first_file_commit
            src_filenames, src_filepaths = get_files_and_path(config[llm_name][language_][key_word]["src"])
            src_filepath = src_filepaths[0]
            src_filename = src_filenames[0]

            print(f'Loading json file-{src_filename}: {src_filepath}')
            json_datas = load_json(src_filepath)
            for index1, data in enumerate(json_datas):
                project_name = data.get('project_name')
                file_commits = data.get('file_all_commit_info_list')
                f_index, f_hash, _, _ = get_hash_first_file(
                    config[llm_name][language_][key_word]["projects"]["src"], project_name, file_commits, key_word)
                if f_index == "":
                    continue
                path_of_first_file_commit_value = Path(project_name) / f_hash / file_commits[f_index].get(
                    'file_path')
                if first_path == str(path_of_first_file_commit_value).replace('\\', '\\'):
                    url_of_first_file_commit = file_commits[f_index].get('html_url')
                    df.at[index, "url_of_first_file_commit"] = url_of_first_file_commit
                    print(df["url_of_first_file_commit"])
                    break


        pass

    if extracted_final_change_code:
        for index, row in df.iterrows():
            language_ = language
            matched_keyword = row['matched_keyword']
            code_language = row['code_language']
            if language_ == "cpp" and code_language == "C":
                language_ = LANGUAGE.C.value
            key_word = matched_keyword.split(" ")[0] if isinstance(matched_keyword, str) else None
            final_chang_commit_path = row['final_change_commit_path']
            if pd.isna(final_chang_commit_path) or not final_chang_commit_path:
                continue
            final_file_path = Path(config[llm_name][language_][key_word]['projects']['src']) / final_chang_commit_path
            start_end_line_data = row['final_start_end_line_data']
            start_end_line_list = parse_str_to_arr(start_end_line_data)
            code_granularity_data = row['code_granularity_data']
            extract_code_and_save(llm_name, language_, key_word, final_file_path, final_chang_commit_path,
                                  start_end_line_list, code_granularity_data, is_save_code=True, is_first=False)

    if cal_complexity == True:
        # Evaluates or redefines a column
        for index, row in df.iterrows():
            if cal_complexity:
                num_code_functions = row["code_functions"]
                num_complexity_all = row["code_complexity_all"]
                num_cognitive_complexity_all = row["code_cognitive_complexity_all"]
                if num_code_functions == 0:
                    df.at[index, "code_complexity_mean_method"] = np.nan
                    df.at[index, "code_cognitive_complexity_mean_method"] = np.nan
                else:
                    df.at[index, "code_complexity_mean_method"] = num_complexity_all*1. / num_code_functions
                    df.at[index, "code_cognitive_complexity_mean_method"] = num_cognitive_complexity_all*1. / num_code_functions




    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print("===========Save File: csv================")
    print("Analysis results have been saved to:", filepath)


if __name__ == "__main__":
    '''
    '''
    # draw_violinplot_manual_vs_gpt(top_percent=0.1)
    # draw_violinplot_from_different_metrics("project_locs", 'Project LOCs Distribution', 'Number of Project LOCs', top_percent=0.1)
    '''

    '''
    # statistic_bug_fix_type(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value)
    # statistic_bug_fix_type(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value)
    # statistic_bug_fix_type(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value)
    # statistic_bug_fix_type(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value)
    # statistic_bug_fix_type(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value)

    '''
    
    '''
    #
    # statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, top_percent=0.1)
    # statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, top_percent=0.1)
    # statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value,top_percent=0.1)
    # statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value,top_percent=0.1)
    # statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, top_percent=0.1)

    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/results-nochange-code-type/python-code-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.Python.value)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/result-code-change-type/python-code-change-type.xlsx",
    #                  LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, is_change=True)
    # statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value)

    # statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value)

    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/results-nochange-code-type/java-code-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.Java.value)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/result-code-change-type/java-code-change-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, is_change=True)
    # statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value)

    statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value)
    statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value)
    statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value)
    statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value)
    statistic_code_function_type(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value)


    '''
    '''
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/results-nochange-code-type/java-code-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.Java.value)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/results-nochange-code-type/code-type-c++.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/results-nochange-code-type/JavaScript-code-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/results-nochange-code-type/TS-code-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value)

    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/result-code-change-type/python-code-change-type.xlsx",
    #                  LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, is_change=True)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/result-code-change-type/java-code-change-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, is_change=True)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/result-code-change-type/C++-code-change-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, is_change=True)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/result-code-change-type/JavaScript-code-change-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, is_change=True)
    # read_xlsx_to_csv("D:\Projects\WhereCodebyGPT/result-code-change-type/TS-code-change-type.xlsx", LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, is_change=True)

    '''
    '''

    # process_final_csv(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, set_url=True)
    # process_final_csv(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, set_url=True)
    # process_final_csv(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, set_url=True)
    # process_final_csv(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, set_url=True)
    # process_final_csv(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, set_url=True)
    # splice_filter_project(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value)

    '''
    '''
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, top_percent=0.1)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, top_percent=0.1)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, top_percent=0.1)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, top_percent=0.1)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, top_percent=0.1)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, only_change=True)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, only_change=True)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, only_change=True)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, only_change=True)
    # calculate_various_measures(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, only_change=True)


    '''
    '''
    # draw_violinplot_from_different_metrics(["star", "contributor", "project_commits", "project_files", "project_locs"], ['Stars Distribution', 'Contributors Distribution', 'Project Commits Distribution',
    #                                                                                                                      'Project Files Distribution', 'Project LOCs Distribution'], ['Number of Stars',
    #                                                                                                                                                                                   'Number of Contributors', 'Number of Project Commits', 'Number of Project Files', 'Number of Project LOCs'])

    # draw_violinplot_from_different_metrics(["star", "project_files", "contributor", "project_commits", "number_of_bug_or_vulnerability_all_commit", "issues"], ['Stars Distribution', 'Project Files Distribution'], ['The number of stars', 'The number of files', "The number of\ncontributors", "The number of commits", "The number of\nbug-fix commits", "The number of issues"], res_name="RQ1")
    # # draw_violinplot_from_different_metrics(["code_locs", "code_complexity_mean_method", "code_cognitive_complexity_mean_method"], ['', '', ""], ['The locs of\ngenerated code', 'The mean cyclomatic\ncomplexity of methods', 'The mean cognitive\ncomplexity of methods'], res_name="RQ2")
    # draw_violinplot_from_different_metrics(["code_locs", "code_complexity_all", "code_cognitive_complexity_all"], ['', '', ""], ['The lines of\ngenerated code', 'The CC of\ngenerated code', 'The CogC of\ngenerated code'], res_name="RQ2")

    # draw_violinplot_from_different_metrics("contributor", 'Contributors Distribution', 'Number of Contributors')
    # draw_violinplot_from_different_metrics("project_commits", 'Project Commits Distribution', 'Number of Project Commits')
    # draw_violinplot_from_different_metrics("project_files", 'Project Files Distribution', 'Number of Project Files')
    # draw_violinplot_from_different_metrics("project_locs", 'Project LOCs Distribution', 'Number of Project LOCs')

    '''
    '''
    # draw_violinplot_manual_vs_gpt()

    '''
    get analysis results
    '''
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Generated.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=False, is_download_project=False, is_sonarqube=True, start=43, end=44)

    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Generated.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Authored.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Coded.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Created.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Written.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Implemented.value)
    # splice_filter_project(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value)

    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Generated.value, is_save_code=True, is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Generated.value, is_save_code=True, is_diff=False, is_download_commit=False, is_download_project=False, is_sonarqube=True, start=184, end=186)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Coded.value, is_save_code=True, is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Created.value, is_save_code=True, is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Written.value, is_save_code=True, is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Written.value, is_save_code=True, is_diff=False, is_download_commit=False, is_download_project=False, is_sonarqube=True, start=40, end=41)


    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Generated.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True, start=358)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Coded.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Written.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)


    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Generated.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)

    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Generated.value, is_save_code=False,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True, start=16, end=17)


    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Created.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Created.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True, start=0, end=1)


    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Written.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)



    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Generated.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)



    '''
    need to run
    '''
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Created.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)


    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Authored.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Created.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Implemented.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Written.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Generated.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)


    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Generated.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True, start=51,
    #                    end=52)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Generated.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True, start=55,
    #                    end=56)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Created.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Written.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)

    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Generated.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True, start=0, end=10)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Authored.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Coded.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Created.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    #
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Written.value, is_save_code=True,
    #                    is_diff=True, is_download_commit=True, is_download_project=True, is_sonarqube=True)
    '''
    download_project
    '''
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Generated.value, is_diff=True, is_download_project=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Authored.value, is_diff=False, is_download_project=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Coded.value, is_diff=False, is_download_project=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Created.value, is_diff=False, is_download_project=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Written.value, is_diff=False, is_download_project=True)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Implemented.value, is_diff=False, is_download_project=True)

    '''
    download commit message
    '''
    # download_commit_message(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Generated.value)

    '''
    only download file
    '''

    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Generated.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Written.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Created.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Authored.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Implemented.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Coded.value)
    #
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Generated.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Written.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Created.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Authored.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Implemented.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.Java.value, KEYWORD.Coded.value)

    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Generated.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Written.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Created.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Authored.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Implemented.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.JavaScript.value, KEYWORD.Coded.value)

    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Generated.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Written.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Created.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Authored.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Implemented.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CPP.value, KEYWORD.Coded.value)

    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Generated.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Written.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Created.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Authored.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Implemented.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.TypeScript.value, KEYWORD.Coded.value)

    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Generated.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Written.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Created.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Authored.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Implemented.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.CSharp.value, KEYWORD.Coded.value)

    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Generated.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Written.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Created.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Authored.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Implemented.value)
    # download_file_only_first(LLM_NAME.ChatGPT.value, LANGUAGE.C.value, KEYWORD.Coded.value)

    '''
    analyze loc change
    '''
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Generated.value, is_diff=False)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Authored.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Coded.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Created.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Written.value)
    # analyze_loc_change(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Implemented.value)

    # request_sonarqube(LLM_NAME.ChatGPT.value, LANGUAGE.Python.value, KEYWORD.Generated_by_ChatGPT.value, is_project=True)
    # print(len(all_columns))
    pass
