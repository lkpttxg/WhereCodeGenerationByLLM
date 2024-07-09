import argparse
import base64
import glob
import json
import os
import time
import urllib.parse
import requests
from tqdm import tqdm
from crawl.search_word_list import VERB_LIST, PREPOSITION_LIST, LLM_LIST, PROGRAM_LANGUAGE
from config import config


def parse_repo_info(url, item):
    """
    get repo info
    :param url:
    :param item:
    :return:
    """
    print(f"get repo info:{url}\n")
    repo_response = requests.get(url, headers=headers)
    repo_data = repo_response.json()
    # 7.start number
    item['repo_star_num'] = repo_data.get('stargazers_count', 0)
    # 8.fork number
    item['repo_fork_num'] = repo_data.get('forks_count', 0)
    # 9.issues number
    item['repo_issues_num'] = repo_data.get('open_issues_count', 0)
    # 10.watch number
    item['repo_watch_num'] = repo_data.get('subscribers_count', 0)

    # 14.main language
    item['repo_main_langauge'] = repo_data.get('language', '')
    # 15.project about
    item['repo_about'] = repo_data.get('description', '')

    # 17.repo create time
    item['repo_create_date'] = repo_data.get('created_at', '')
    # 18.repo last update time
    item['repo_update_date'] = repo_data.get('updated_at', '')
    # 19.get repo tag
    item['repo_tags'] = repo_data.get('topics', '')


def parse_repo_all_commit(url, item, owner, repo_name):
    """
    parse repo all commit info
    :param url:
    :param item:
    :return:
    """
    commit_info_list = item['repo_all_commit_info_list']
    print(f"get repo all commit info{url}\n")
    while True:
        repo_all_commit_resp = requests.get(url, headers=headers)
        ra_data = repo_all_commit_resp.json()
        if isinstance(ra_data, list):
            for commit in ra_data:
                temp_dict = {}
                temp_dict['hash_code'] = commit['sha']
                detail_info = commit['commit']
                temp_dict['description'] = detail_info['message']
                temp_dict['date'] = detail_info['author']['date']
                temp_dict['link'] = detail_info['url']
                temp_dict['html_url'] = commit['html_url']
                temp_dict[
                    'project_download_url'] = f"https://api.github.com/repos/{owner}/{repo_name}/zipball/{commit['sha']}"
                commit_info_list.append(temp_dict)
        elif isinstance(ra_data, dict):
            print(url)
            for commit in ra_data['items']:
                temp_dict = {}
                temp_dict['hash_code'] = commit['sha']
                detail_info = commit['commit']
                temp_dict['description'] = detail_info['message']
                temp_dict['date'] = detail_info['author']['date']
                temp_dict['link'] = detail_info['url']
                temp_dict['html_url'] = commit['html_url']
                temp_dict[
                    'project_download_url'] = f"https://api.github.com/repos/{owner}/{repo_name}/zipball/{commit['sha']}"

                commit_info_list.append(temp_dict)
        link = str(repo_all_commit_resp.headers.get('link', None))
        if link is not None:
            larr = link.split(",")[0]
            if "next" in larr:
                next_url = larr.split("; ")[0].replace("b'<", "").replace(">", "").replace("<", "")
                url = next_url
            else:
                break
        else:
            break


def parse_file_commit_detail_info(url, temp_dict):
    """
    get file commit detail info
    :param url:
    :param temp_dict:
    :return:
    """
    print(f"get file commit detail info{url}\n")
    detail_resp = requests.get(url, headers=headers)
    detail_data = detail_resp.json()

    temp_dict['size'] = detail_data.get('size', '')
    temp_dict['download_url'] = detail_data.get('download_url', '')
    temp_dict['html_url'] = detail_data.get('html_url', '')
    temp_dict['content'] = decode_base64_content(detail_data['content'])


def decode_base64_content(content):
    """
    decode base64 content
    :param content:
    :return:
    """
    utf8_line_arr = []
    base64_lines = content.split("\n")
    for line in base64_lines:
        try:
            decoded_bytes = base64.b64decode(line)
            decoded_string = decoded_bytes.decode('utf-8')
            utf8_line_arr.append(decoded_string)
        except Exception:
            utf8_line_arr.append("")
    combined_string = "\n".join(utf8_line_arr)
    return combined_string


def parse_file_all_commit(url, item, file_path, owner, repo_name):
    """
    parse file all commit
    :param url:
    :param item:
    :param file_path:
    :param owner:
    :param repo_name:
    :return:
    """
    commit_info_list = item['file_all_commit_info_list']
    print("get file all commit info\n")
    while True:
        file_commit_resp = requests.get(url, headers=headers)
        fc_data = file_commit_resp.json()
        if isinstance(fc_data, list):
            for commit in fc_data:
                temp_dict = {}
                temp_dict['hash_code'] = commit['sha']
                temp_dict['file_path'] = file_path
                temp_dict['create_date'] = commit['commit']['author']['date']
                temp_dict[
                    'project_download_url'] = f"https://api.github.com/repos/{owner}/{repo_name}/zipball/{commit['sha']}"
                detail_url = f'https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}?ref={commit["sha"]}'
                parse_file_commit_detail_info(url=detail_url, temp_dict=temp_dict)
                commit_info_list.append(temp_dict)
        elif isinstance(fc_data, dict):
            for commit in fc_data['items']:
                temp_dict = {}
                temp_dict['hash_code'] = commit['sha']
                temp_dict['file_path'] = file_path
                temp_dict['create_date'] = commit['commit']['author']['date']
                temp_dict[
                    'project_download_url'] = f"https://api.github.com/repos/{owner}/{repo_name}/zipball/{commit['sha']}"
                detail_url = f'https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}?ref={commit["sha"]}'
                parse_file_commit_detail_info(url=detail_url, temp_dict=temp_dict)
                commit_info_list.append(temp_dict)
        link = str(file_commit_resp.headers.get('link', None))
        print(link)
        if link is not None:
            larr = link.split(",")[0]
            if "next" in larr:
                next_url = larr.split("; ")[0].replace("b'<", "").replace(">", "").replace("<", "")
                url = next_url
            else:
                break
        else:
            break


def generate_github_search_url(keyword, language, page):
    BASE_URL = "https://api.github.com/search/code?q="
    encoded_keyword = urllib.parse.quote(keyword)

    query = f"{encoded_keyword}+in:file+language:{urllib.parse.quote(language)}&per_page={100}&page={page}"
    full_url = f"{BASE_URL}{query}"
    return full_url


def file_json_formatter(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        json_str = file.read()
    json_str = '[' + json_str

    last_comma_index = json_str.rfind(',')
    result = json_str[:last_comma_index] + ']'

    with open(file_path, "w", encoding="utf8") as file:
        file.write(result)


headers = {}


def crawl_action(github_token,page_wait_time,info_wait_time):
    global headers
    headers = {
        'Accept': 'application/vnd.github.text-match+json',
        'Authorization': f'Bearer {github_token}',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    for language in PROGRAM_LANGUAGE:
        for verb in VERB_LIST:
            # data save path
            data_save_path = config["chatgpt"][language][verb]['src']
            url_list = []
            keyword_list = []
            # get all combination list
            for preposition in PREPOSITION_LIST:
                for llm in LLM_LIST:
                    # TODO: add \" means exact search if not means fuzzy search
                    keyword_list.append(f"\"{verb} {preposition} {llm}\"")
            # get all combination corresponding url
            for index, keyword in enumerate(keyword_list):
                print(f"parsed keyword:{keyword}  process {index + 1}/{len(keyword_list) + 1}")
                url = generate_github_search_url(keyword, language, page=1)
                url_list.append(url)
                response = requests.get(url=url, headers=headers)
                time.sleep(page_wait_time)
                resp_data = response.json()
                total_number = resp_data['total_count']
                print(total_number)
                for i in range(0, (total_number // 100) + 1):
                    req_url = generate_github_search_url(keyword, language, i + 1)
                    url_list.append(req_url)

            for url in url_list:
                print(f"----------------{language}----------------{url}----------------")
                with tqdm(desc="processing items", unit="item") as pbar:
                    time.sleep(info_wait_time)
                    response = requests.get(url, headers=headers)
                    resp_data = response.json()
                    try:
                        if resp_data['total_count'] == 0:
                            continue
                    except Exception:
                        continue
                    for index, item in enumerate(resp_data['items']):
                        try:
                            code_item_info_map = {}
                            # get repo info
                            repo_info = item['repository']
                            # get project name
                            code_item_info_map['project_name'] = repo_info.get('name', '')
                            # get project html info
                            code_item_info_map['project_html_url'] = repo_info.get('html_url', '')
                            print("============================")
                            print(f"get repo {repo_info.get('name', '')}abstract info:{url}\n")
                            # get project html url
                            code_item_info_map['contributor_num'] = ''
                            code_item_info_map['language_info'] = ''
                            REPO_INFO_BASE_URL = f"https://api.github.com/repos/{repo_info['full_name']}"
                            parse_repo_info(url=REPO_INFO_BASE_URL,
                                            item=code_item_info_map)
                            owner = repo_info['owner']['login']
                            repo_name = repo_info['name']
                            REPO_COMMIT_BASE_URL = f"https://api.github.com/repos/{owner}/{repo_name}/commits?per_page=100&page=1"
                            code_item_info_map['repo_all_commit_info_list'] = []
                            parse_repo_all_commit(url=REPO_COMMIT_BASE_URL, item=code_item_info_map, owner=owner,
                                                  repo_name=repo_name)

                            file_path = item['path']
                            code_item_info_map['file_all_commit_info_list'] = []
                            FILE_COMMIT_BASE_URL = f"https://api.github.com/repos/{owner}/{repo_name}/commits?path={file_path}&per_page=100&page=1"
                            parse_file_all_commit(url=FILE_COMMIT_BASE_URL, item=code_item_info_map,
                                                  file_path=file_path,
                                                  owner=owner, repo_name=repo_name)
                            with open(f"{data_save_path}/{language}_data.json", "a+",
                                      encoding='utf8') as file:
                                json_str = json.dumps(code_item_info_map, ensure_ascii=False)
                                file.write(json_str + "," + "\n")
                                file.flush()
                        except Exception as e:
                            print(e)
                            with open(f"{data_save_path}/url_failed.txt", "a+", encoding="utf8") as file:
                                file.write(f"{language}#{url}#{index + 1}\n")
                                file.flush()
                                continue
                        with open(f"{data_save_path}/url_success.txt", "a+", encoding='utf8') as file:
                            file.write(f"{language}#{url}#{index + 1}\n")
                            file.flush()
                        print("============================")
                        pbar.update(1)
            file_json_formatter(f"{data_save_path}/{language}_data.json")
        print("all crawl finished")
