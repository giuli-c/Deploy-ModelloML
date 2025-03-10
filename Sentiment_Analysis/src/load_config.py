import yaml
import os

class LoadConfig:
    def __init__():
        self.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def load_config():
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
        except FilenNotFoundError as e:
            print(f"Errore: Il file {config_path} non è stato trovato.")
        except yaml.YAMLError as e:
            print(f"Errore nella lettura del file YAML: {e}")
        except Exception as e:
            print(f"Errore: {e}")
        return config