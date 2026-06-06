from __future__ import annotations

from typing import Any


class DomainCatalogService:
    """Centralized domain catalog synchronization and validation."""

    def __init__(self, cfg: dict[str, Any], domain_map: dict[str, Any], runtime_state_module: Any):
        self.cfg = cfg
        self.domain_map = domain_map
        self.runtime_state = runtime_state_module

    def sync_domains(self) -> set[str]:
        """
        Build active domains from config + registry + runtime state.
        Also ensures domain_map has entries for runtime-added domains.
        """
        active_domains = {
            domain for domain, settings in self.cfg["domains"].items()
            if settings.get("enabled")
        }

        # Include domains from registry (if available)
        try:
            from utils.registry import load_registry
            registry = load_registry()
            active_domains.update(registry.get("domains", {}).keys())
        except Exception:
            pass

        runtime_domains = self.runtime_state.get_runtime_domains()
        for domain_name, details in runtime_domains.items():
            active_domains.add(domain_name)
            if domain_name not in self.domain_map:
                self.domain_map[domain_name] = {
                    "files": [],
                    "description": details.get("description", f"{domain_name} (runtime domain)"),
                    "keywords": details.get("keywords", []),
                }
            else:
                if details.get("description"):
                    self.domain_map[domain_name]["description"] = details["description"]
                if details.get("keywords"):
                    self.domain_map[domain_name]["keywords"] = details["keywords"]

        return active_domains
