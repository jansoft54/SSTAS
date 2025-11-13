from pathlib import Path
from yaml_configuration import Configuration

class ConfigManager:
   

    def __init__(self):
        self.path = Path("./config.yml")
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")
        self._cfg = Configuration(self.path)

    def __getattr__(self, name):
        if hasattr(self._cfg, name):
            return getattr(self._cfg, name)
        raise AttributeError(f"'ConfigManager' object has no attribute '{name}'")

    def __getitem__(self, key):
        return getattr(self._cfg, key)

    def __repr__(self):
        return repr(self._cfg)
