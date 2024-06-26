import os
from pathlib import Path
import yaml

# define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def get_root(loader, node):
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")


# register the tag handler
yaml.add_constructor('!join', join)
yaml.add_constructor('!root', get_root)


root_path = os.path.dirname(os.path.abspath(__file__))

with open(f'{root_path}/path.yml', 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)


def create_directories_from_config(config, base_path=''):
    if isinstance(config, dict):
        for key, value in config.items():
            create_directories_from_config(value, base_path)
    elif isinstance(config, str) and config.startswith(base_path):
        Path(config).mkdir(parents=True, exist_ok=True)


create_directories_from_config(config)


if __name__ == "__main__":
    print(config)
