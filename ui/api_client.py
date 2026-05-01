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
    
    def upload_standard_document(self, uploaded_file):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }
        return self._post("/upload", files=files, timeout=300)
    
    def upload_graphrag_document(self, uploaded_file):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }
        return self._post("/graphrag/upload", files=files, timeout=600)
    
    def generate_single(self, payload: dict):
        return self._post("/generate", json=payload, timeout=300)

    def generate_multi(self, payload: dict):
        return self._post("/multi/generate", json=payload, timeout=300)

    def generate_graphrag(self, payload: dict):
        return self._post("/graphrag/generate", json=payload, timeout=300)
    
    def rebuild_multi_index(self):
        return self._post("/multi/rebuild", timeout=900)

    def rebuild_graphrag_domain(self, domain_name: str):
        return self._post(f"/graphrag/rebuild/{domain_name}", timeout=900)





