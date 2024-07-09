import argparse
import base64
import glob
import json
import os
import time
import urllib.parse
from pathlib import Path
import requests
from tqdm import tqdm
from crawl.search_word_list import VERB_LIST, PREPOSITION_LIST, LLM_LIST, PROGRAM_LANGUAGE


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


def dir_file_json_formatter(dir_path):
    file_list = glob.glob(f"{dir}/*_data.json")
    print(file_list)
    for file_path in file_list:
        with open(file_path, "r", encoding="utf8") as file:
            json_str = file.read()
        json_str = '[' + json_str

        last_comma_index = json_str.rfind(',')
        result = json_str[:last_comma_index] + ']'

        with open(file_path, "w", encoding="utf8") as file:
            file.write(result)


def parse_args():
    """
    parse args
    :return:
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', type=str, help='file dir list', required=True)
    parser.add_argument('-t', type=str, help='token str', required=True)
    args = parser.parse_args()
    return args


headers = {}


def crawl_action(github_token):
    global headers
    headers = {
        'Accept': 'application/vnd.github.text-match+json',
        'Authorization': f'Bearer {github_token}',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    base_directory_path = "./data"
    # ================= url generate ====================
    print("start generate crawl url")
    for verb in VERB_LIST:
        for preposition in PREPOSITION_LIST:
            for llm in LLM_LIST:
                search_key_word = f"\"{verb} {preposition} {llm}\""
                dir_key_word_path = search_key_word.replace('\"', "").replace(" ", "-")
                dir = Path(f"./data/{dir_key_word_path}")
                if not dir.exists():
                    dir.mkdir()
                for language in PROGRAM_LANGUAGE:
                    print(language)
                    url = generate_github_search_url(search_key_word, language, 1)
                    print(url)
                    response = requests.get(url=url, headers=headers)
                    time.sleep(8)
                    resp_data = response.json()
                    total_number = resp_data['total_count']
                    print(total_number)
                    for i in range(0, (total_number // 100) + 1):
                        req_url = generate_github_search_url(search_key_word, language, i + 1)
                        print(req_url)
                        with open(f"{base_directory_path}/{dir_key_word_path}/url.txt", "w", encoding='utf8') as file:
                            file.write(f"{language}#{req_url}\n")
                            file.flush()
    print("generate crawl url end")
    # ================= url generate ====================

    # ================= start crawl =====================
    print("start crawl ")
    folder_names = [name for name in os.listdir(base_directory_path) if
                    os.path.isdir(os.path.join(base_directory_path, name))]
    for keyword in folder_names:
        with open(f"{base_directory_path}/{keyword}/url.txt", "r", encoding="utf8") as file:
            url_lines = file.readlines()
        # parse url get language and url
        for url in url_lines:
            url_info = url.split("\#")
            language = url_info[0]
            req_url = url_info[1].strip()
            print(f"----------------{language}----------------{req_url}----------------")
            pre_language = language
            with tqdm(desc="processing items", unit="item") as pbar:
                time.sleep(0.4)
                response = requests.get(req_url, headers=headers)
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
                        print(f"get repo {repo_info.get('name', '')}abstract info:{req_url}\n")
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
                        with open(f"{base_directory_path}/{keyword}/{language}_data.json", "a+",
                                  encoding='utf8') as file:
                            json_str = json.dumps(code_item_info_map, ensure_ascii=False)
                            file.write(json_str + "," + "\n")
                            file.flush()
                    except Exception as e:
                        print(e)
                        with open(f"{base_directory_path}/{keyword}/url_failed.txt", "a+", encoding="utf8") as file:
                            file.write(f"{language}#{req_url}#{index + 1}\n")
                            file.flush()
                            continue
                    with open(f"{base_directory_path}/{keyword}/url_success.txt", "a+", encoding='utf8') as file:
                        file.write(f"{language}#{url}#{index + 1}\n")
                        file.flush()
                    print("============================")
                    pbar.update(1)
        dir_file_json_formatter(f"{base_directory_path}/{keyword}")
    print("all crawl finished")
