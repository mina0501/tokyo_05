import yaml

def get_config(key: str):
    with open('src/hot_config.yml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config.get(key, None)
