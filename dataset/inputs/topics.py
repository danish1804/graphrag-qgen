# ============================================================
# dataset/inputs/topics.py
# Domain topic definitions for question dataset generation.
#
# Each domain has a list of topics used to generate learning
# objectives for the input CSV.
#
# Author: Mohammed Danish Alam
# Project: P000233DS — EdGenAI Technologies Capstone
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

# ── Cross-domain topics for medium questions ──────────────
# 15 topics per domain pair — no duplicates across pairs

CROSS_DOMAIN_TOPICS = {

    ("deep_learning", "database"): [
        "Machine learning model storage and retrieval",
        "Feature engineering and data preprocessing",
        "Training data management at scale",
        "Vector databases for embedding storage",
        "Data versioning for machine learning pipelines",
        "Query optimisation for training data retrieval",
        "Database indexing for large scale datasets",
        "Transaction management in ML feature stores",
        "Schema design for machine learning metadata",
        "Caching strategies for training data pipelines",
        "Data normalisation for neural network inputs",
        "Distributed storage for deep learning datasets",
        "SQL vs NoSQL for machine learning applications",
        "Database backup and recovery for model checkpoints",
        "Concurrency control in data preprocessing pipelines",
    ],

    ("deep_learning", "software_engineering"): [
        "MLOps and model deployment pipelines",
        "Testing strategies for machine learning systems",
        "Version control for machine learning experiments",
        "Continuous integration for deep learning projects",
        "Software design patterns for neural network systems",
        "Code modularity in deep learning frameworks",
        "Debugging and validation of neural network training",
        "Requirements engineering for AI systems",
        "Technical debt in machine learning codebases",
        "Refactoring deep learning pipelines",
        "Agile development for machine learning projects",
        "Documentation standards for AI model code",
        "Risk management in machine learning deployments",
        "Software metrics for evaluating ML system quality",
        "Safety critical considerations in AI software",
    ],

    ("deep_learning", "operating_systems"): [
        "GPU memory management for deep learning",
        "Parallel processing in neural network training",
        "Resource scheduling for machine learning workloads",
        "Process management for distributed training jobs",
        "Virtual memory constraints in large model training",
        "CPU and GPU synchronisation in deep learning",
        "File system optimisation for dataset loading",
        "Thread management in data loading pipelines",
        "Deadlock prevention in parallel training systems",
        "Memory mapped files for large dataset access",
        "OS scheduling impact on neural network training",
        "Inter-process communication in distributed training",
        "System calls used in deep learning frameworks",
        "Context switching overhead in model training",
        "Real time constraints in neural network inference",
    ],

    ("deep_learning", "computer_architecture"): [
        "Hardware acceleration for neural networks",
        "Memory bandwidth constraints in deep learning",
        "Parallelism in convolutional neural network computation",
        "Cache efficiency in matrix multiplication operations",
        "Instruction pipelining for neural network inference",
        "SIMD instructions for deep learning operations",
        "Memory hierarchy optimisation for model training",
        "Branch prediction impact on neural network code",
        "Floating point precision in neural network training",
        "GPU architecture and its suitability for deep learning",
        "Amdahl's law applied to neural network parallelisation",
        "DMA transfers in deep learning data pipelines",
        "RISC architecture advantages for inference workloads",
        "Out of order execution in neural network computations",
        "Power consumption considerations in AI hardware",
    ],

    ("database", "software_engineering"): [
        "Object relational mapping and database abstraction",
        "Database driven application architecture",
        "Testing database dependent systems",
        "Database schema migration in agile projects",
        "Version control strategies for database schemas",
        "Design patterns for database access layers",
        "Requirements engineering for database systems",
        "Continuous integration with database testing",
        "Refactoring database interactions in legacy systems",
        "Software metrics for database application quality",
        "Risk management in database migration projects",
        "Transaction handling in object oriented applications",
        "Unit testing strategies for database operations",
        "Configuration management for database environments",
        "Code review practices for SQL and database code",
    ],

    ("database", "operating_systems"): [
        "Transaction management and process synchronisation",
        "Deadlock prevention in database and OS contexts",
        "File system and database storage management",
        "Buffer pool management in database systems",
        "OS level locking vs database level locking",
        "Virtual memory impact on database performance",
        "Process scheduling for database query execution",
        "Semaphores in database concurrency control",
        "File system choice for database storage",
        "Memory management for database buffer pools",
        "Inter-process communication in database servers",
        "Page replacement algorithms in database systems",
        "Disk scheduling impact on database I/O performance",
        "System calls used in database management systems",
        "Real time database systems and OS scheduling",
    ],

    ("database", "computer_architecture"): [
        "Database query performance and memory hierarchy",
        "Disk storage architecture and database indexing",
        "Cache efficiency in database systems",
        "B-tree indexing and memory access patterns",
        "Instruction pipelining in database query engines",
        "RISC architecture benefits for database workloads",
        "DMA transfers in database I/O operations",
        "Multicore processors for parallel query execution",
        "Floating point arithmetic in database aggregations",
        "Bus architecture impact on database throughput",
        "Memory bandwidth and database join operations",
        "Cache coherence in shared memory database systems",
        "Out of order execution in database query processing",
        "Virtual memory and database buffer management",
        "Amdahl's law applied to parallel database queries",
    ],

    ("software_engineering", "operating_systems"): [
        "Concurrent programming and thread safety",
        "System software design and OS interfaces",
        "Process management in software applications",
        "Synchronisation mechanisms in concurrent software",
        "Deadlock avoidance in multi-threaded applications",
        "File system APIs in software engineering",
        "Memory management in application development",
        "Inter-process communication in software design",
        "Real time software development for OS environments",
        "System call design and software architecture",
        "Testing concurrent and parallel software systems",
        "Agile development for system software projects",
        "Configuration management for OS dependent software",
        "Risk management in concurrent system development",
        "Performance engineering for OS interfacing code",
    ],

    ("software_engineering", "computer_architecture"): [
        "Low level optimisation and compiler design",
        "Embedded systems software development",
        "Performance engineering and profiling",
        "Software design for cache friendly data structures",
        "Assembly language integration in software projects",
        "Branch prediction impact on software performance",
        "RISC architecture considerations in software design",
        "Memory layout optimisation in software engineering",
        "Instruction set architecture and compiler design",
        "Testing strategies for hardware software interfaces",
        "Agile development for embedded systems",
        "Requirements engineering for hardware constrained systems",
        "Version control for firmware and embedded code",
        "Technical debt in low level software systems",
        "Code review for performance critical software",
    ],

    ("operating_systems", "computer_architecture"): [
        "Virtual memory implementation and hardware support",
        "Interrupt handling and processor architecture",
        "Cache coherence in multiprocessor systems",
        "TLB management and address translation hardware",
        "DMA controller interaction with OS drivers",
        "Instruction pipelining and OS context switching",
        "Memory management unit design and OS support",
        "Multicore scheduling and hardware thread management",
        "Bus arbitration and OS I/O management",
        "Floating point exceptions and OS signal handling",
        "Branch prediction and OS kernel performance",
        "RISC architecture benefits for OS kernel design",
        "Out of order execution and OS memory consistency",
        "Hardware interrupts and OS interrupt service routines",
        "Power management in OS and hardware co-design",
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
    total_cross = sum(len(t) for t in CROSS_DOMAIN_TOPICS.values())
    print(f"Total cross-domain topics: {total_cross}")
    print(f"Hard multi-domain topics: {len(HARD_MULTI_DOMAIN_TOPICS)}")
    print(f"Case study topics: {len(CASE_STUDY_TOPICS)}")