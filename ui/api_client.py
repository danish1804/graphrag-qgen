import requests

DEFAULT_API_BASE_URL = "http://localhost:8000"


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, timeout: int = 30):
        response = requests.get(f"{self.base_url}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, json=None, files=None, timeout: int = 300):
        response = requests.post(
            f"{self.base_url}{path}",
            json=json,
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def health(self):
        return self._get("/health", timeout=10)
    
    def multi_status(self):
        return self._get("/multi/status", timeout=10)

    def graphrag_status(self):
        return self._get("/graphrag/status", timeout=10)

