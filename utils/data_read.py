import json
# from .path_utils import get_files_and_path


def parse_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    def print_structure(data, indent=0):
        if isinstance(data, dict):
            for key, value in data.items():
                print('  ' * indent + str(key) + ':')
                print_structure(value, indent + 1)
        elif isinstance(data, list):
            for index, item in enumerate(data):
                print('  ' * indent + f'[{index}]:')
                print_structure(item, indent + 1)
        else:
            print('  ' * indent + str(data))

    print_structure(data)


def load_jsonl_mul(file_path, is_arr=False, is_json=False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if is_arr:
                data = f.read().strip()
                if not data.startswith('['):
                    data = '[' + data
                if data.endswith(','):
                    data = data[:-1]
                    data += ']'
                data = json.loads(data)
                print("==========Read File: load json===========")
                print("file_path:" + file_path)
                # data = [line for line in data]
            elif is_json:
                data = f.read()
                data = json.loads(data)
                print("==========Read File: load json===========")
                print("file_path:" + file_path)
                data = [(k, v) for (k, v) in data.items()]
            else:
                data = f.readlines()
                print("==========Read File: load json===========")
                print("file_path:" + file_path)
                data = [json.loads(line) for line in data]
    except json.JSONDecodeError as e:
        print("JSON parsing error：", e.msg)
        print("Error position (number of characters)：", e.pos)
        error_context = 10
        start = max(e.pos - error_context, 0)
        end = min(e.pos + error_context, len(data))
        print("Error near the content：", data[start:end])
    return data


def load_json(file_path):
    return load_jsonl_mul(file_path, is_arr=True)




if __name__ == "__main__":
    pass

