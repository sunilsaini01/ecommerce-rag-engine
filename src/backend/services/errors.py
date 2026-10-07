class DependencyUnavailable(Exception):
    """Public-safe dependency error; underlying connection details stay private."""

    def __init__(self, service: str):
        self.service = service
        super().__init__(f"{service} is temporarily unavailable. Please try again shortly.")
