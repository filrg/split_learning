import logging
import os


class Colors:
    COLORS = {
        "header": '\033[95m',
        "blue": '\033[94m',
        "green": '\033[92m',
        "yellow": '\033[93m',
        "red": '\033[91m',
        "end": '\033[0m'
    }


class Logger:
    def __init__(self, log_path, debug_mode=False, minimal=False):
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.DEBUG)  # Mức log
        self.debug_mode = debug_mode
        self.minimal = minimal

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

    def log_info(self, message):
        if not self.minimal:
            print(f"[INFO] {message}")
        self.logger.info(message)

    def log_warning(self, message):
        print_with_color(f"[WARN] {message}", "yellow")
        self.logger.warning(message)

    def log_error(self, message):
        print_with_color(f"[ERROR] {message}", "red")
        self.logger.error(message)

    def log_debug(self, message):
        if self.debug_mode:
            print_with_color(f"[DEBUG] {message}", "green")
            self.logger.debug(message)


def print_with_color(text, color):
    color_code = Colors.COLORS.get(color.lower(), Colors.COLORS["end"])
    print(f"{color_code}{text}{Colors.COLORS['end']}")
