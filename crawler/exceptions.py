class DomainMismatchException(Exception):
    """Exception raised for URLs whose domain does not match the news website's domain."""

    def __init__(
        self,
        url: str,
        message: str = "URL's domain does not match the news website's domain",
    ):
        self.url = url
        self.message = message
        super().__init__(self.message)


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail

    def __str__(self) -> str:
        return f"HTTPException: {self.status_code} - {self.detail}"
