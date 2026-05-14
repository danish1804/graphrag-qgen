# ============================================================
# dataset/inputs/topics.py
# Domain topic definitions for question dataset generation.
#
# Each domain has a list of topics used to generate learning
# objectives for the input CSV.
#
# To add new topics: append to the relevant domain list.
# To add a new domain: add a new key to DOMAIN_TOPICS.
#
# Author: Mohammed Danish Alam
# Project: P000233DS — EdGenAI Technologies Capstone
# ============================================================

# ── Domain Topics ─────────────────────────────────────────
# Each topic is a short noun-phrase (NOT a question).
# These are used as learning objectives passed to the
# GraphRAG generation system.
# ============================================================

DOMAIN_TOPICS = {

    "deep_learning": [
        "Convolutional neural networks",
        "Dropout regularisation",
        "Backpropagation and gradient descent",
        "Recurrent neural networks and sequence modelling",
        "Transformer architecture and attention mechanism",
        "Batch normalisation",
        "Activation functions in neural networks",
        "Transfer learning and fine-tuning",
        "Pooling layers and spatial downsampling",
        "Loss functions and optimisation",
        "Overfitting and regularisation techniques",
        "Vanishing and exploding gradient problem",
        "Autoencoders and representation learning",
        "Generative adversarial networks",
        "Neural network weight initialisation",
    ],

    "database": [
        "Database normalisation",
        "SQL joins — inner and outer",
        "ACID properties in database transactions",
        "Transaction management and concurrency",
        "Indexing and query optimisation",
        "Two phase locking and deadlock prevention",
        "Entity relationship modelling",
        "Relational algebra and calculus",
        "Stored procedures and triggers",
        "Database recovery and logging",
        "NoSQL vs relational databases",
        "B-tree and hash indexing",
        "Query execution plans and optimiser",
        "Database security and access control",
        "Data warehousing and OLAP",
    ],

    "software_engineering": [
        "Software development lifecycle models",
        "Agile methodology and requirements management",
        "Verification and validation in software testing",
        "Unit testing and integration testing",
        "Test driven development",
        "Design patterns in object oriented programming",
        "Software architecture and modularity",
        "Version control and configuration management",
        "Code review and static analysis",
        "Risk management in software projects",
        "Requirements engineering and specification",
        "Software metrics and quality assurance",
        "Refactoring and technical debt",
        "Continuous integration and deployment",
        "Safety critical software testing",
    ],

    "operating_systems": [
        "Processes and threads",
        "CPU scheduling algorithms",
        "Virtual memory and paging",
        "Deadlock detection and prevention",
        "Banker algorithm and resource allocation",
        "Memory management and segmentation",
        "File systems and directory structures",
        "Interprocess communication",
        "Semaphores and mutual exclusion",
        "Disk scheduling algorithms",
        "Page replacement algorithms",
        "System calls and kernel mode",
        "Context switching and process states",
        "Real time operating systems",
        "Input output management",
    ],

    "computer_architecture": [
        "Instruction pipelining",
        "Cache memory hierarchy",
        "Pipeline hazards and resolution techniques",
        "Memory hierarchy and locality",
        "Instruction set architecture",
        "Superscalar and out of order execution",
        "Branch prediction",
        "RISC vs CISC architecture",
        "Virtual memory and address translation",
        "Multicore processors and parallelism",
        "Amdahl's law and performance limits",
        "Direct memory access",
        "Bus architecture and interconnects",
        "Floating point representation",
        "Assembly language and low level programming",
    ],
}

# ── Cross-domain topics for medium/hard questions ─────────
# These are used when generating multi-domain questions.
# The objective spans multiple domains naturally.

CROSS_DOMAIN_TOPICS = {

    ("deep_learning", "database"): [
        "Machine learning model storage and retrieval",
        "Feature engineering and data preprocessing",
        "Training data management at scale",
    ],

    ("deep_learning", "software_engineering"): [
        "MLOps and model deployment pipelines",
        "Testing strategies for machine learning systems",
        "Version control for machine learning experiments",
    ],

    ("deep_learning", "operating_systems"): [
        "GPU memory management for deep learning",
        "Parallel processing in neural network training",
        "Resource scheduling for machine learning workloads",
    ],

    ("deep_learning", "computer_architecture"): [
        "Hardware acceleration for neural networks",
        "Memory bandwidth constraints in deep learning",
        "Parallelism in convolutional neural network computation",
    ],

    ("database", "software_engineering"): [
        "Object relational mapping and database abstraction",
        "Database driven application architecture",
        "Testing database dependent systems",
    ],

    ("database", "operating_systems"): [
        "Transaction management and process synchronisation",
        "Deadlock prevention in database and OS contexts",
        "File system and database storage management",
    ],

    ("database", "computer_architecture"): [
        "Database query performance and memory hierarchy",
        "Disk storage architecture and database indexing",
        "Cache efficiency in database systems",
    ],

    ("software_engineering", "operating_systems"): [
        "Concurrent programming and thread safety",
        "System software design and OS interfaces",
        "Process management in software applications",
    ],

    ("software_engineering", "computer_architecture"): [
        "Low level optimisation and compiler design",
        "Embedded systems software development",
        "Performance engineering and profiling",
    ],

    ("operating_systems", "computer_architecture"): [
        "Virtual memory implementation and hardware support",
        "Interrupt handling and processor architecture",
        "Cache coherence in multiprocessor systems",
    ],
}

# ── Hard multi-domain topics (3+ domains) ─────────────────
HARD_MULTI_DOMAIN_TOPICS = [
    "Deadlock prevention across OS databases and distributed systems",
    "Performance optimisation across hardware software and data layers",
    "Memory management from hardware cache to virtual memory to database buffer",
    "Concurrency control in operating systems databases and multi-threaded applications",
    "System design integrating architecture OS and software engineering principles",
    "Resource scheduling across CPU memory and database query execution",
    "Security mechanisms across OS software and database layers",
    "Fault tolerance and recovery across system layers",
    "Scalability from hardware architecture to software design to database tuning",
    "Data flow from CPU registers through memory hierarchy to persistent storage",
    "Pipeline optimisation across CPU instruction pipelines and software processing",
    "Testing strategies from unit tests to system integration and hardware validation",
    "Virtualisation across hardware OS and database layers",
    "Distributed systems design across networking OS and database domains",
    "Real time constraints across hardware scheduling and software design",
]

# ── Case study topics ──────────────────────────────────────
CASE_STUDY_TOPICS = [
    "A university database system experiencing slow query performance",
    "A neural network model showing signs of overfitting on training data",
    "An operating system experiencing frequent deadlocks in a multi-user environment",
    "A software project failing to meet deadlines due to requirement changes",
    "A CPU-intensive application running slowly on a pipelined processor",
    "A bank transaction system with concurrency and data integrity issues",
    "A machine learning pipeline with data preprocessing bottlenecks",
    "A web application with database connection pool exhaustion",
    "An embedded system with memory constraints and real-time requirements",
    "A distributed database system experiencing network partition failures",
    "A legacy software system requiring refactoring and modernisation",
    "A deep learning model with vanishing gradients during training",
    "An OS scheduler causing starvation for low priority processes",
    "A computer architecture design with cache miss rate issues",
    "A software team transitioning from waterfall to agile methodology",
]


if __name__ == "__main__":
    print("Domain Topics Summary:")
    print("=" * 50)
    total = 0
    for domain, topics in DOMAIN_TOPICS.items():
        print(f"  {domain}: {len(topics)} topics")
        total += len(topics)
    print(f"\nTotal single-domain topics: {total}")
    print(f"Cross-domain pairs: {len(CROSS_DOMAIN_TOPICS)}")
    print(f"Hard multi-domain topics: {len(HARD_MULTI_DOMAIN_TOPICS)}")
    print(f"Case study topics: {len(CASE_STUDY_TOPICS)}")