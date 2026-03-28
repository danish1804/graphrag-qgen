import torch

print("=" * 45)
print("GPU CHECK")
print("=" * 45)
print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM             : {vram:.1f} GB")
    print(f"CUDA version     : {torch.version.cuda}")

print("\n" + "=" * 45)
print("PACKAGE CHECK")
print("=" * 45)

packages = [
    ("transformers",          "transformers"),
    ("peft",                  "peft"),
    ("trl",                   "trl"),
    ("bitsandbytes",          "bitsandbytes"),
    ("sentence-transformers", "sentence_transformers"),
    ("faiss",                 "faiss"),
    ("langchain",             "langchain"),
    ("langchain-community",   "langchain_community"),
    ("pypdf",                 "pypdf"),
    ("python-pptx",           "pptx"),
    ("fastapi",               "fastapi"),
    ("datasets",              "datasets"),
    ("accelerate",            "accelerate"),
    ("python-dotenv",         "dotenv"),
    ("huggingface_hub",       "huggingface_hub"),
    ("tqdm",                  "tqdm"),
    ("numpy",                 "numpy"),
    ("pydantic",              "pydantic"),
]

all_good = True
for display, imp in packages:
    try:
        mod = __import__(imp)
        version = getattr(mod, "__version__", "unknown")
        print(f"  OK   {display:<25} {version}")
    except ImportError as e:
        print(f"  FAIL {display:<25} {e}")
        all_good = False

print("\n" + "=" * 45)
if all_good and torch.cuda.is_available():
    print("ALL GOOD — environment ready")
else:
    print("ISSUES FOUND — paste output here")
print("=" * 45)