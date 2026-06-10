from src.debug.log import get_logger

class Config:
    REQUIRED_KEYS = ['EV3_HOST', 'EV3_PORT']

    def __init__(self):
        self.config = self.load_config()
        self.check_required_values()

    def load_config(self) -> dict:
        config = {}
        with open('controller.config', 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    get_logger().warning(f"Warning: Invalid line {line_num} (will be ignored): {line}")
                    continue
                try:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
                except Exception as e:
                    get_logger().error(f"Error parsing line {line_num}: {e}")
        return config
    
    def check_required_values(self) -> None:
        missing = [key for key in self.REQUIRED_KEYS if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")
        
    def getStr(self, key: str) -> str:
        if key not in self.config:
            raise ValueError(f"Key does not exist in config: {key}")
        return self.config[key]
    
    def getNum(self, key: str) -> int:
        if key not in self.config:
            raise ValueError(f"Key does not exist in config: {key}")
        return int(self.config[key])
