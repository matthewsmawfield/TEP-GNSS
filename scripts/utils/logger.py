import logging
import sys
from pathlib import Path

# Define the root path of the project for relative path calculations
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

class TEPLogger:
    def __init__(self, name: str = "tep_gnss", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level(level))
        
        # Prevent adding multiple handlers if the logger already exists
        if not self.logger.handlers:
            # Create a console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(self._get_log_level(level))
            
            # Create formatter
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            ch.setFormatter(formatter)
            
            self.logger.addHandler(ch)
            self.logger.propagate = False # Prevent messages from being passed to the root logger

    def _get_log_level(self, level_name: str):
        return getattr(logging, level_name.upper(), logging.INFO)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)
        
    def process(self, message: str):
        self.logger.info(f"[PROCESSING] {message}")

    def success(self, message: str):
        self.logger.info(f"[SUCCESS] {message}")
        
    def test(self, message: str):
        self.logger.info(f"[TEST] {message}")

# Global logger instance
logger = TEPLogger().logger
