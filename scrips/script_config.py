from pathlib import Path
import json


CONFIG_PATH = Path(__file__).resolve().with_name("script_config.json")


def get_data_file():
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    data_file = config.get("DATA_FILE")
    if not isinstance(data_file, str) or not data_file:
        raise ValueError(f"Invalid DATA_FILE in {CONFIG_PATH}: {data_file!r}")

    return data_file
