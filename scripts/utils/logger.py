import logging
import sys
from pathlib import Path

# Define the root path of the project for relative path calculations
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

class TEPFormatter(logging.Formatter):
    """Formatter with color support that matches step 3's print_status format."""

    # ANSI color codes matching step 3
    COLORS = {
        'SUCCESS': '\033[1;32m',  # Green bold
        'WARNING': '\033[1;33m',  # Yellow bold
        'ERROR': '\033[1;31m',    # Red bold
        'INFO': '\033[0;37m',     # White
        'DEBUG': '\033[0;90m',    # Dark gray
        'PROCESS': '\033[0;34m',  # Blue
        'TEST': '\033[1;35m'      # Magenta bold
    }
    RESET = '\033[0m'

    def __init__(self, fmt=None, datefmt=None):
        # Use same timestamp format as step 3: %H:%M:%S
        super().__init__(fmt, datefmt='%H:%M:%S')

    def format(self, record):
        message = record.getMessage()

        # Map log levels to display names and colors
        level_mapping = {
            25: ('PROCESS', self.COLORS['PROCESS']),   # Custom PROCESS level
            26: ('SUCCESS', self.COLORS['SUCCESS']),  # Custom SUCCESS level
            27: ('TEST', self.COLORS['TEST']),        # Custom TEST level
            logging.INFO: ('INFO', self.COLORS['INFO']),
            logging.WARNING: ('WARNING', self.COLORS['WARNING']),
            logging.ERROR: ('ERROR', self.COLORS['ERROR']),
            logging.DEBUG: ('DEBUG', self.COLORS['DEBUG'])
        }

        level_name, color = level_mapping.get(record.levelno, ('INFO', self.COLORS['INFO']))

        # Format exactly like step 3: [timestamp] [LEVEL] message
        timestamp = self.formatTime(record, self.datefmt)
        return f"{color}[{timestamp}] [{level_name}] {message}{self.RESET}"

class TEPLogger:
    def __init__(self, name: str = "tep_gnss", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level(level))
        
        # Prevent adding multiple handlers if the logger already exists
        if not self.logger.handlers:
            # Create a console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)  # Allow all custom levels

            # Create formatter
            formatter = TEPFormatter()
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
        # Use a custom logging level for PROCESS messages
        self.logger.log(25, message)  # 25 is between INFO(20) and WARNING(30)

    def success(self, message: str):
        # Use a custom logging level for SUCCESS messages
        self.logger.log(26, message)  # 26 is between INFO(20) and WARNING(30)

    def test(self, message: str):
        # Use a custom logging level for TEST messages
        self.logger.log(27, message)  # 27 is between INFO(20) and WARNING(30)

    def debug_msg(self, message: str):
        self.logger.debug(message)

# Global logger instance
logger = TEPLogger().logger
