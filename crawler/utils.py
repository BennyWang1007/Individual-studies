
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

    _instances: dict = {}

    def __new__(cls, *args, **kwargs):
        if args[0] not in cls._instances:
            cls._instances[args[0]] = super(Logger, cls).__new__(cls)
        return cls._instances[args[0]]

    def __init__(self, name, verbose_level: int = 1):
        self.name = name
        self.verbose_level = verbose_level

    def get_logger(self, name: str | None = None):
        if name:
            return Logger(name)
        return self

    def set_verbose_level(self, level: int):
        if level < 0 or level > 2:
            raise ValueError("Verbose level must be between 0 and 2.")
        self.verbose_level = level

    def __log(self, method, message, color1: str = "", color2: str = ""):
        print(f"{color1}[{self.name}] [{method}] {message}{color2}")

    def info(self, message):
        if self.verbose_level >= 1:
            self.__log("INFO", message)

    def debug(self, message):
        if self.verbose_level >= 2:
            self.__log("DEBUG", message, TERM_COLORS.YELLOW, TERM_COLORS.ENDC)

    def error(self, message):
        if self.verbose_level >= 0:
            self.__log("ERROR", message, TERM_COLORS.RED, TERM_COLORS.ENDC)
