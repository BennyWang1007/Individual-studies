import threading
from datetime import datetime


class TERM_COLORS:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    ENDC = "\033[0m"


class Logger:
    """
    Verbose logger class for logging messages with different verbosity levels.

    This class is a singleton, meaning that only one instance of it can exist
    for each unique name. The logger can be used to log messages at different
    verbosity levels, allowing for easy filtering of log messages based on the
    verbosity level set. The verbosity levels range from 0 (no output) to 4
    (all messages including debug). The logger also supports colored output
    for different log levels.

    The logger can also write logs to a file with the format:
    [time] [name] [method] [message]

    A write lock is used to prevent concurrent writes to the log file.

    Usage:

        logger = Logger("my_logger", verbose_level=3, log_file="log.txt")

        logger.info("This is an info message.")

        logger.error("This is an error message.")

        logger.warning("This is a warning message.")

        logger.debug("This is a debug message.")

        logger.log("This is a custom log message.", method="CUSTOM",\
                   color=TERM_COLORS.GREEN)

    Attributes:
        name (str): The name of the logger instance.
        verbose_level (int): The verbosity level of the logger. (default: 3)

            0: No output

            1: Error messages only

            2: Error and warning messages

            3: Error, warning, and info messages

            4: All messages (including debug)

        log_file (str, optional): Path to the log file. If provided, logs will\
            be written to this file. If not provided, it will be defaulted to\
            "log.txt". The log file will be created if it does not exist.
    """

    _instances: dict = {}
    _write_lock: threading.Lock = threading.Lock()
    _active_log_files: set[str] = set()

    def __new__(cls, *args, **kwargs):
        if args[0] not in cls._instances:
            cls._instances[args[0]] = super(Logger, cls).__new__(cls)
        return cls._instances[args[0]]

    def __init__(
        self, name, verbose_level: int = 3, log_file: str = "log.txt"
    ):
        if not hasattr(self, 'initialized'):
            self.name = name
            self.verbose_level = verbose_level
            self.log_file = log_file
            # add 4 new lines to the log file if it's not used already
            if log_file not in Logger._active_log_files:
                Logger._active_log_files.add(log_file)
                self.__write_log("\n\n\n\n")
            self.initialized = True

    def get_logger(self, name: str | None = None):
        if name:
            return Logger(name)
        return self

    def set_verbose_level(self, level: int):
        """
        Set the verbosity level of the logger. (default: 3)

        0: No output

        1: Error messages only

        2: Error and warning messages

        3: Error, warning, and info messages

        4: All messages (including debug)
        """
        if level < 0 or level > 4:
            raise ValueError("Verbose level must be between 0 and 4.")
        self.verbose_level = level

    def __write_log(self, message):
        with Logger._write_lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message)
            except Exception as e:
                print(
                    f"{TERM_COLORS.RED}[ERROR] Failed to write to log file: "
                    f"{e}{TERM_COLORS.ENDC}"
                )
                raise e

    def __log(self, method, message, color1: str = "", color2: str = ""):
        print(f"{color1}[{self.name}] [{method}] {message}{color2}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{self.name}] [{method}] {message}\n"
        self.__write_log(log_entry)

    def error(self, message):
        if self.verbose_level > 0:
            self.__log("ERROR", message, TERM_COLORS.RED, TERM_COLORS.ENDC)

    def warning(self, message):
        if self.verbose_level > 1:
            self.__log("WARN", message, TERM_COLORS.YELLOW, TERM_COLORS.ENDC)

    def info(self, message):
        if self.verbose_level > 2:
            self.__log("INFO", message)

    def log(self, message, method="LOG", color=""):
        if self.verbose_level > 2:
            self.__log(method, message, color, TERM_COLORS.ENDC)

    def debug(self, message):
        if self.verbose_level > 3:
            self.__log("DEBUG", message, TERM_COLORS.BLUE, TERM_COLORS.ENDC)
