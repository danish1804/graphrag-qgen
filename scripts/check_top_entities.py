import pickle
import os
import networkx as nx

BASE_DIR   = os.getenv( "GRAPHRAG_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

domains = [
    "software_engineering",
    "operating_systems",
    "database",
    "deep_learning",
    "computer_architecture"
]

for domain in domains:
    path = os.path.join(
        BASE_DIR, "indexes", f"graphrag_{domain}_graph.pkl")

    if not os.path.exists(path):
        print(f"\n{domain}: not built yet — skipping")
        continue

    with open(path, "rb") as f:
        g = pickle.load(f)

    degrees = sorted(
        g.degree(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    print(f"\n{domain} — top 15 entities:")
    print("-" * 55)
    for entity, degree in degrees:
        print(f"  {entity:<45} degree={degree}")