import os
import json

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

OPEN_ROUTER_API_KEY = config['OPEN_ROUTER_API_KEY']
MODEL_NAME = config['MODEL_NAME']