# ============================================================
# src/graphrag_domain_indexer.py
# Per-domain GraphRAG indexer
#
# Builds separate knowledge graphs per subject domain.
# Routes queries to the correct domain graph at generation time.
#
# Why per-domain graphs:
#   "pooling" in Deep Learning ≠ "pooling" in Databases
#   "transaction" in OS ≠ "transaction" in Databases
#   Mixed corpus corrupts entity relationships
#   Domain-specific graphs produce accurate retrieval
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.graphrag_indexer import build_graphrag_index
from src.graphrag_indexer import load_graphrag_index
from src.document_loader  import load_and_chunk
import numpy as np

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = r"D:\Semester 4\AI_Postr_Graduate_Project\graphrag_qgen"
TESTED_DIR = os.path.join(BASE_DIR, "data", "tested")
INDEX_DIR  = os.path.join(BASE_DIR, "indexes")

# ── Domain definitions ────────────────────────────────────
DOMAIN_MAP = {
    "deep_learning": {
        "files": [
            "source_50.pdf",
            "source_51.pdf",
            "source_52.pdf",
            "source_53.pdf",
            "source_54.pdf",
            "source_55.pdf",
            "source_56.pdf",
            "source_57.pdf",
            "source_58.pdf",
            "source_59.pdf",
            "source_60.pdf",
            "source_61.pdf",
            "source_62.pdf",

        ],
        "description": "Deep Learning — CNNs, pooling, backprop",
        "keywords": [
            "cnn", "convolutional", "neural network", "pooling",
            "backpropagation", "gradient", "deep learning",
            "convolution", "relu", "activation", "batch",
            "epoch", "training", "feature map", "kernel",
            "stride", "padding", "weight", "bias", "layer",
            "softmax", "dropout", "overfitting", "regularisation", "weight sharing"
        ]
    },
    "database": {
        "files": [
            "source_7.pdf",
            "source_8.pdf",
            "source_13.pdf",
            "source_14.pdf",
            "source_17.pdf",
            "source_18.pdf",
            "source_31.pdf",
            "source_36.pdf",
            "source_49.pdf",
        ],
        "description": "Database Systems — SQL, transactions, indexing, design",
        "keywords": [
            "sql", "database", "query", "transaction", "relation",
            "table", "index", "schema", "normalisation", "join",
            "primary key", "foreign key", "acid", "concurrency",
            "deadlock", "hash table", "b-tree", "indexing",
            "tuple", "attribute", "entity", "relational",
            "dbms", "er diagram", "integrity", "constraint",
            "functional dependency", "decomposition", "algebra"
        ]
    },
    "software_engineering": {
        "files": [
            "source_1.pdf",
            "source_2pdf",
            "source_3.pdf",
            "source_4.pdf",
            "source_11.pdf",
            "source_15.pdf",
            "source_16.pdf",
            "source_21.pdf",
            "source_22.pdf",
            "source_37.pdf",
            "source_45.pdf",
            "source_46.pdf",
            "source_47.pdf",
            "source_48.pdf",

        ],
        "description": "Software Engineering — processes, patterns, testing",
        "keywords": [
            "software", "agile", "design pattern", "testing",
            "requirement", "architecture", "scrum", "sprint",
            "refactoring", "oop", "inheritance", "polymorphism",
            "abstraction", "encapsulation", "uml", "use case",
            "waterfall", "prototype", "class diagram", "module",
            "interface", "component", "deployment", "maintenance",
            "build", "version control", "safety", "process",
            "object oriented", "object-oriented", "class",
            "unit test", "integration test", "software quality",
            "software safety", "hazard", "reliability"
        ]
    },
    "operating_systems": {
        "files": [
            "source_5.pdf",    
            "source_10.pdf",   
            "source_19.pdf",   
            "source_20.pdf",   
            "source_33.pdf",   
            "source_35.pdf",   
            "source_39.pdf",   
            "source_40.pdf",   
            "source_41.pdf", 
            "source_42.pdf",   
            "source_43.pdf", 
        ],
        "description": "Operating Systems — processes, memory, scheduling",
        "keywords": [
            "process", "thread", "scheduler", "memory",
            "virtual memory", "page", "segment", "kernel",
            "interrupt", "system call", "file system", "disk",
            "cache", "buffer", "semaphore", "mutex", "deadlock",
            "unix", "linux", "fork", "pipe", "signal", "socket",
            "i/o", "device driver", "ipc", "context switch",
            "scheduling algorithm", "round robin", "priority",
            "xv6", "trap", "page table", "tlb"
        ]
    },
    "computer_architecture": {
        "files": [
            "source_6.pdf",    
            "source_9.pdf",    
            "source_12.pdf",   
            "source_23.pdf",   
            "source_24.pdf",   
            "source_25.pdf",   
            "source_26.pdf",   
            "source_27.pdf",   
            "source_28.pdf",   
            "source_29.pdf",   
            "source_30.pdf",   
            "source_32.pdf",   
            "source_38.pdf",   
            "source_44.pdf",   
        ],
        "description": "Computer Architecture — CPU, memory hierarchy, instruction sets, pipelining",
        "keywords": [
            "cpu", "instruction set", "pipeline", "cache",
            "memory hierarchy", "register", "alu", "risc", "cisc",
            "branch prediction", "out of order", "superscalar",
            "amdahl", "addressing mode", "interrupt", "dma",
            "bus", "clock cycle", "cpi", "throughput", "hazard",
            "forwarding", "stall", "fetch", "decode", "execute",
            "writeback", "direct mapped", "associative", "tlb",
            "virtual address", "physical address", "cache hit",
            "cache miss", "write back", "write through"
        ]
    }

}
