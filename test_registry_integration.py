# test_registry_integration.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.registry import get_domain_files

print("Testing registry integration...")

for domain in ["deep_learning", "database",
               "software_engineering",
               "operating_systems",
               "computer_architecture"]:
    files = get_domain_files(domain)
    print(f"{domain}: {len(files)} files")

# Check source_63 is included
dl_files = get_domain_files("deep_learning")
print(f"\nsource_63.pdf in deep_learning: "
      f"{'source_63.pdf' in dl_files}")