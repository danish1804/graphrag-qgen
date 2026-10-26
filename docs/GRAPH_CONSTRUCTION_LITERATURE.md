# Graph construction for GraphRAG question generation: literature review and plan

**Author:** Mohammed Danish Alam · **Date:** 8 October 2026 · **Requested by:** Samar Shailendra
**Purpose:** find the landmark GraphRAG and educational-knowledge-graph work, compare it with how
our graph is built, and turn it into an evidence-based plan for improving the graph and measuring
whether it is better.

Every paper below was fetched and checked. **PR** = peer-reviewed, **PP** = preprint. Items we
could not verify are listed at the end and should not be cited yet.

---

## 1. Headline findings

1. **Our design already has a name: Microsoft's FastGraphRAG.** Noun phrases from spaCy,
   co-occurrence edges. Microsoft itself calls this graph "quite a bit noisier" than the LLM-built
   one, but by default it weights edges with **PMI** (pointwise mutual information) and prunes
   nodes by frequency and degree. We use raw counts and no pruning. That difference alone explains
   why "time", "problem" and "three" are among deadlock's strongest neighbours.
2. **Our neighbour-expansion step may be hurting retrieval.** In HippoRAG's ablation (NeurIPS 2024),
   "seed nodes plus neighbours" scored *below* "seed nodes only" (average R@2 42.2 vs 50.7).
   Personalised PageRank over the graph scored highest (57.4). We expand to neighbours.
3. **Semantic (embedding-similarity) edges help.** HippoRAG adds "synonym" edges between concepts
   whose embeddings are close; removing them drops R@5 from 72.9 to 70.5. NodeRAG reports
   46.29% vs 41.71% without them. These are the edges that would connect our isolated concepts.
4. **On CS textbooks, graph methods barely beat keyword search.** GraphRAG-Bench on 20 CS textbooks
   (Xiao et al., PP): BM25 71.66, GraphRAG 72.50, HippoRAG 72.64; two graph methods fell below no
   retrieval at all. Any claim that our graph helps must be tested against BM25 and plain dense
   retrieval, not only against our previous graph.
5. **There is no landmark education-specific GraphRAG paper yet.** The established education work is
   concept-map and prerequisite extraction from textbooks (Penn State, Tsinghua) and ontology-based
   question generation (Manchester). GraphRAG for tertiary IT assessment with a modern LLM is an
   open gap, which is a credible contribution for our paper.
6. **Bloom templates only separate the bottom level.** With real students (Demaidi et al. 2017),
   Remember items were easier than the rest, but Understand, Apply and Analyse were not
   distinguishable from each other. "More hops = harder" is not established (a 2026 preprint found
   hop count added nothing to difficulty prediction). Relation type (is-a vs property) *did* change
   difficulty.
7. **There is a robust way to measure graph quality that is not an LLM judge.** The textbook's own
   back-of-book index is an expert-made concept list (Alpizar-Chacon & Sosnovsky 2021), and edge
   accuracy can be estimated from a small stratified sample with two annotators (Gao et al., VLDB
   2019). Combined with downstream retrieval and question quality, this answers Samar's open
   question.

---

## 2. How the landmark methods build their graphs

| Method | Citation | Extraction | Edges | Node handling | Query-time use | Takeaway for us |
|---|---|---|---|---|---|---|
| **GraphRAG** | Edge et al. 2024/25, PP. [arXiv 2404.16130](https://arxiv.org/abs/2404.16130) | LLM (GPT-4-turbo) on 600-token chunks, re-prompted ("gleaning") for missed entities | Typed relations with descriptions; weight = times extracted | Exact-string merge; hierarchical Leiden communities with LLM summaries | Global map-reduce over community summaries; local entity neighbourhood | Leiden communities need no LLM; useful for topics and diversity |
| **FastGraphRAG** | Microsoft GraphRAG docs and code (not peer-reviewed). [docs](https://microsoft.github.io/graphrag/index/methods/) | spaCy/NLTK noun phrases | Co-occurrence in a text unit; **PMI-normalised weights by default**; small 50–100-token units recommended | Pruning by min frequency, frequency and degree outliers, low-weight edges | Leiden communities | **Closest to ours. Adopt PMI weighting and pruning directly** |
| **LightRAG** | Guo et al., Findings of EMNLP 2025, PR. [arXiv 2410.05779](https://arxiv.org/abs/2410.05779) | LLM entities and relations, LLM-written node and edge profiles | Typed | Deduplication | Keyword matching to entities (low level) and relations (high level), plus one hop | Description-heavy; fails with small models (see MiniRAG) |
| **HippoRAG** | Gutiérrez et al., NeurIPS 2024, PR. [arXiv 2405.14831](https://arxiv.org/abs/2405.14831) | Two-step LLM (entities, then triples) | Extracted triples **plus synonym edges at embedding cosine ≥ 0.8** | **Node specificity** = 1 / passages containing the node | Personalised PageRank from query entities | Synonym edges, node specificity and PPR all need no LLM |
| **HippoRAG 2** | Gutiérrez et al., ICML 2025, PR. [arXiv 2502.14802](https://arxiv.org/abs/2502.14802) | As above | Triples, synonym edges, and **passage nodes** linked to their phrases | Synonym edges outnumber extracted ones about 8 to 1 | PPR seeded with phrases and passages | Matching the query to *nodes* (what we do) scored 59.6 R@5 vs 87.1 matching to triples |
| **NodeRAG** | Xu et al. 2025, PP. [arXiv 2504.11544](https://arxiv.org/abs/2504.11544) | LLM decomposition into 7 node types | Typed + **HNSW semantic edges** | k-core and betweenness to pick key nodes | Exact + vector search, shallow PPR | Semantic kNN edges; shallow PPR works better than deep |
| **KAG** | Liang et al., WWW '25 Companion, PR. [arXiv 2409.13731](https://arxiv.org/abs/2409.13731) | Schema-constrained + free LLM extraction | Adds **synonym, is-a and part-of** alignment edges | Merges fragmented entities | Logical-form planner | Is-a / part-of edges and synonym alignment |
| **MiniRAG** | Fan et al., ACL 2026, PR. [arXiv 2501.06713](https://arxiv.org/abs/2501.06713) | Named entities + chunk nodes (built for small models) | Entity–chunk links | — | Topology-based search | Warning: with 1.5–4B models GraphRAG failed and LightRAG fell below naive RAG |
| **PathRAG** | Chen et al., AAAI 2026, PR. [arXiv 2502.14902](https://arxiv.org/abs/2502.14902) | LightRAG-style | Typed | — | Flow-based pruning to key paths | Argues the problem is redundant retrieval, not missing retrieval |
| **G-Retriever** | He et al., NeurIPS 2024, PR. [arXiv 2402.07630](https://arxiv.org/abs/2402.07630) | None (assumes an existing graph) | — | — | Steiner-tree subgraph + GNN | Fell below no retrieval on CS textbooks |
| Survey | Peng et al., ACM TOIS 2025, PR. [arXiv 2408.08921](https://arxiv.org/abs/2408.08921) | Indexing → retrieval → generation framework | | | | Framing for the paper |

### Evidence on specific construction choices

| Choice | Evidence | Source |
|---|---|---|
| LLM vs classical extraction | HippoRAG avg R@5: GPT-3.5 72.9, Llama-3.1-8B 67.8, REBEL (non-LLM) 58.4. spaCy dependency triples reached 94% of LLM extraction on enterprise data (weak, LLM-judged) | HippoRAG; Min et al. 2025, PP |
| Small models | 1.5–4B models break LLM-built graph pipelines; 7B sits just above that range, so keep any extraction schema small | MiniRAG |
| Co-occurrence vs typed | **No head-to-head comparison found.** Triple-only graphs contain only about 65.8% of answer entities, so keep the source text attached | Han et al., KDD 2026, PR |
| PMI weighting | NPMI is bounded and less frequency-sensitive; used by default in FastGraphRAG; **no paper measures PMI vs raw counts inside GraphRAG**, so this would be a new result | Bouma 2009; Microsoft code |
| Semantic edges | −2.4 R@5 (HippoRAG) and −4.6 accuracy (NodeRAG) without them. The 0.8 threshold was tuned for other embedders; ours (bge-large) needs its own threshold | HippoRAG; NodeRAG |
| Neighbour expansion vs PPR | Neighbours 42.2 < seeds only 50.7 < PPR 57.4 (avg R@2) | HippoRAG |
| When graphs help | Plain RAG matches or beats GraphRAG on simple fact retrieval; GraphRAG wins on multi-step reasoning and summarisation | GraphRAG-Bench, Xiang et al., ICLR 2026, PR |

---

## 3. Education-specific work

| Work | Citation | What they build | Relations | Evaluation | Use for us |
|---|---|---|---|---|---|
| **Concept maps from textbooks** | Wang et al., CIKM 2016, PR. [MSR](https://www.microsoft.com/en-us/research/publication/using-prerequisites-extract-concept-maps-textbooks/) | Key concepts and prerequisite links from 6 textbooks (one is databases) | Prerequisite (directed) | 3 graduate annotators per domain; relation F1 0.52–0.62 with textbook features vs 0.17–0.44 for a baseline | **Table-of-contents distance and chapter co-occurrence as prerequisite signals**, computable from our PDFs |
| Concept hierarchy | Wang et al., DocEng 2015, PR. [PSU](https://pure.psu.edu/en/publications/concept-hierarchy-extraction-from-textbooks/) | Chapter-level concept hierarchy | Chapter → concept | Manual hierarchies, 3 textbooks | Chapter membership as structure |
| **Knowledge models from PDF textbooks** | Alpizar-Chacon & Sosnovsky 2021, New Rev. Hypermedia & Multimedia, PR. [link](https://orion.tec.ac.cr/en/publications/knowledge-models-from-pdf-textbooks/) | TOC, back-of-book index, formatting | Structure, index terms | "High accuracy" | **The index is a free expert-made gold concept list** |
| Index term specificity | Alpizar-Chacon & Sosnovsky, WWW '22, PR (abstract via search only) | Core / in-domain / related / foreign index terms | Domain specificity | — | Tag nodes by domain specificity; may help routing |
| KnowEdu | Chen et al. 2018, IEEE Access, PR. [DOAJ](https://doaj.org/article/00fe8cc0d06a440181fc1d1be7a5a830) | Maths concept graph | Prerequisite | Two experts, κ 0.945 (concepts), 0.896 (relations) | Annotation protocol: binary labels, accept only on agreement, report κ |
| RefD | Liang et al., EMNLP 2015, PR. [ACL](https://aclanthology.org/D15-1193/) | Prerequisites from asymmetric references | Prerequisite | Beats supervised baselines, 7 domains | Reference asymmetry over our own chunks (untested adaptation) |
| Prerequisites in MOOCs | Pan et al., ACL 2017, PR. [ACL](https://aclanthology.org/P17-1133/) | Prerequisites from course videos | Prerequisite | **Annotator κ 0.59–0.65**; F1 about 69–72% | Realistic agreement target for prerequisite labels |
| LectureBank | Li et al., AAAI 2019, PR. [arXiv 1811.12181](https://arxiv.org/abs/1811.12181) | 1,352 lecture files, 208 topics | Prerequisite | Cohen's κ 0.70 | Built from slides, close to our corpus |
| Active learning for prerequisites | Liang et al., AAAI 2018, PR. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11396) | Label uncertain pairs first | Prerequisite | 4 textbook domains | Cuts annotation effort |
| MOOCCube / MOOCCubeX | Yu et al., ACL 2020 / CIKM 2021, PR | Large MOOC concept datasets (Chinese) | Prerequisite and others | Dataset papers | Background only |
| LLM-built EduKG | Jhajj et al., ITS 2024, PR | GPT-4 triples from course material and objectives | LLM-proposed | Auto + human | Same pattern with local Qwen, human-checked |

### Graph-based question generation and difficulty

| Work | Citation | Finding | Use for us |
|---|---|---|---|
| **Ontology-based questions, real students** | Demaidi, Gaber & Filer 2017, Smart Learning Environments, PR. [link](https://slejournal.springeropen.com/articles/10.1186/s40561-017-0046-6) | Computer-networks ontology; 88–126 students per test; CTT and IRT. **Is-a items easier and more discriminating than property items** (p < 0.05). **Bloom: Remember easier than the rest; no difference among Understand, Apply, Analyse** | The evaluation model for our paper: an IT domain, real students, psychometrics |
| Predicting difficulty from the graph | Kurdi et al. 2021, Semantic Web, PR. [link](https://semantic-web-journal.net/node/2380) | Ontology measure 47% accuracy vs experts 49% | Graph features predict difficulty about as well as experts |
| **Concept map as prompt context** | Scaria et al., ECAI 2025, PR. [arXiv 2505.02850](https://arxiv.org/abs/2505.02850) | 75.2% of items met all expert criteria vs about 37% for plain LLM and plain RAG | Pass structured graph context, not only chunks |
| Graph RAG in MOOCs | Abdelmagied, Chatti et al., EMOOCs 2025, PR. [arXiv 2505.10074](https://arxiv.org/abs/2505.10074) | Slide-level graph; question quality 2.86/3; QA accuracy only 45%, failing on ambiguous names | Similar to our setup; entity disambiguation matters |
| Hop count and difficulty | Şakiroğlu et al. 2026, PP. [arXiv 2604.10748](https://arxiv.org/abs/2604.10748) | 156 items, ~38 responses each; R² 0.52; **removing hop count did not hurt prediction** | Do not assume more hops means harder; test it |
| Path depth | Amanlou et al. 2026 (KNIGHT), PP. [arXiv 2602.20135](https://arxiv.org/abs/2602.20135) | Depth-controlled items; r ≈ 0.78 with error rate, few data points, self-validated | Weak support only |
| Systematic review | Kurdi et al. 2020, IJAIED, PR. [DOI](https://doi.org/10.1007/s40593-019-00186-y) | "Little focus on generating questions of controlled difficulty" | Background for the gap |

---

## 4. Measuring whether a graph is better

The proposal is four levels, fixed before any graph change:

| Level | Measure | Source | Notes |
|---|---|---|---|
| Structure | Share of non-isolated nodes, average degree, clustering, largest-component share, hub concentration | GraphRAG-Bench (Xiao; Xiang) | Reference points for non-isolated share: GraphRAG 72.5%, LightRAG 69.7%, HippoRAG 89.6%; ours is 61–72%. Necessary but not sufficient: synonym edges inflate these |
| **Concept coverage** | Node precision and recall against the textbook index, glossary and contents headings | Alpizar-Chacon & Sosnovsky 2021; Wang et al. 2015/16 | Expert-made, independent of us, no annotation cost |
| **Edge accuracy** | Stratified sample of 150–200 edges per domain; two annotators; binary labels; accuracy with a 95% margin; Cohen's κ | Gao et al., PVLDB 2019, PR; KnowEdu; Pan 2017 | Expect κ about 0.6–0.7 for prerequisite edges. Add a must-link list (for example deadlock–mutex) and measure how many are connected within 1–2 hops (our design) |
| Coherence | NPMI of neighbourhoods on a held-out reference corpus | Lau et al., EACL 2014, PR | Circular if computed on the same corpus as NPMI-weighted edges |
| **Downstream** | Retrieval recall on the gold set for graph variants **and BM25 and dense-only baselines**; then question quality | GraphRAG-Bench | A graph is better only if correctness or coverage improves and downstream holds |

---

## 5. Plan for the two pilot subjects (operating systems, databases)

Ordered by evidence and cost. Each variant is built in a separate index folder so production is
untouched.

| # | Change | Evidence | Effort |
|---|---|---|---|
| 0 | **Measurement first:** index-based gold concept list, edge sample sheet, structural baseline, BM25 and dense baselines | Section 4 | 1–2 days |
| 1 | **Extraction cleanup:** drop numeric entities ("three", "four"), person names and page ranges; fix the singulariser ("databases" → "databas") | Audit findings | 0.5 day |
| 2 | **NPMI edge weights, node specificity, pruning** of very frequent and high-degree generic nodes | FastGraphRAG defaults; HippoRAG node-specificity ablation | 1 day, no LLM |
| 3 | **Semantic kNN edges** for concepts (mutual nearest neighbours above a threshold calibrated for bge-large) | HippoRAG, NodeRAG | 1 day |
| 4 | **Retrieval over the graph:** compare current neighbour expansion with seeds-only and Personalised PageRank (with chunk nodes) | HippoRAG, HippoRAG 2 | 1–2 days |
| 5 | **Typed, directed edges:** prerequisite from table-of-contents distance and reference asymmetry; is-a / part-of from patterns; later, a small relation schema with local Qwen, validated by humans | Wang 2016; Liang 2015; KAG; Graphusion relation set (unverified venue) | 3–5 days |
| 6 | **Use the graph in generation:** pass the typed subgraph as structured context; log graph features for every question | Scaria 2025; Demaidi 2017 | 2–3 days |

Steps 0–4 fit this week. Steps 5–6 are the research contribution and follow once the measurement
shows which graph variant is best per subject, which also tests Samar's point that different
subjects may need different graphs.

---

## 6. Not verified (do not cite yet)

Graphusion and Min et al. (venues unconfirmed, treat as preprints); Xue & Zou KG-quality survey
(TKDE 2022, seen in search only); Olney et al. 2012; Manrique et al. 2019; Alsubait "lessons learnt"
figures; KAQG IRT results; evaluation numbers for Aytekin & Saygın 2024, Jhajj et al. 2024 and
Alpizar-Chacon & Sosnovsky (abstracts give none).
