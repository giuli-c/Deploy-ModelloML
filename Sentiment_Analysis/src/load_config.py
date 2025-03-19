import yaml
import os

class LoadConfig:
    def __init__(self, config_file="config.yaml"):
        self.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", config_file))

    def load_config(self):
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError as e:
            print(f"Errore: Il file {self.config_path} non è stato trovato.")
        except yaml.YAMLError as e:
            print(f"Errore nella lettura del file YAML: {e}")
        except Exception as e:
            print(f"Errore: {e}")
        return config