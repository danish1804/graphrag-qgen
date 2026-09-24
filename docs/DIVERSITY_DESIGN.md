# Generating multiple distinct questions from one query

**Project:** P000233DS (EdGenAI) · **Author:** Mohammed Danish Alam · **17 September 2026**
**Status:** research-backed design for Samar's diversity request. Not yet implemented.

---

## 1. The problem

Retrieval is deterministic. The same learning outcome returns the same chunks, so asking
for three questions from one outcome gives three near-identical questions. This blocks the
learning-outcome work, where a single compound CLO legitimately needs several questions.

**The proposed mechanism** was: record which graph nodes were used for question 1, then
bias question 2 toward nodes not yet visited, tracking "percentage of nodes visited" as a
coverage measure.

**The objection raised:** if we push toward unvisited nodes, do we drift away from what is
actually relevant to the query?

**The answer: yes, that failure is real and has been measured — and the mechanism as
stated is the weakest design in this literature.** But there are six separable fixes, only
one of which is a tuning parameter.

---

## 2. The objection is empirically justified

Zhai, Cohen and Lafferty measured it directly: a novelty-heavy setting produced
**−8.1% S-precision**. The mechanism they name is the important part —

> relevant documents overlap each other more than non-relevant documents do, so penalising
> overlap systematically promotes junk.

That is precisely the concern, confirmed. Two relevant chunks about paging resemble each
other far more than a paging chunk resembles an unrelated one, so "reward dissimilarity"
is partly "reward irrelevance".

Compounding it, Cuconasu et al. (SIGIR 2024) found accumulated semantically-related-but-
wrong chunks cost up to **−67%** answer accuracy, a single distractor up to **−25%**, with
a sweet spot of 3–5 retrieved documents. **Our own k sweep found the same shape** —
coverage plateaus at k=7 while precision falls — so this is not a hypothetical risk here.

---

## 3. Six ways the literature prevents drift

| # | Mechanism | Source | Fits our graph? |
|---|---|---|---|
| 1 | **Diversify over a relevance-filtered pool; never re-retrieve** | MMR's `θ` threshold | **Yes — highest value, lowest cost** |
| 2 | λ bounds how much relevance you will pay | MMR | Yes |
| 3 | Relevance multiplies *inside* the diversity term, so an irrelevant item contributes ~0 | xQuAD, PM-2 | Yes |
| 4 | Monotone submodular coverage — reward coverage additively instead of penalising redundancy | Lin & Bilmes (2011) | Yes, cleanest theoretically |
| 5 | Random walk stays anchored by a relevance prior in the teleport vector | GRASSHOPPER | **Yes — direct fit, we have the graph** |
| 6 | Relevance floor: do not diversify when the pool is thin | PM-2 failure analysis | Yes |

### 3.1 The one change worth making first

**MMR never retrieves. It reorders a pool that relevance already fenced.** Carbonell and
Goldstein define the candidate set as the documents retrieved above *"a relevance threshold
θ, below which it will not retrieve documents"*. Off-topic drift is bounded **by
construction**, not by tuning.

LangChain implements this as over-fetch then rerank: `fetch_k=20`, `k=4`,
`lambda_mult=0.5` — a **5× over-fetch ratio**. PM-2 used ~2.5×; Yigit-Sert et al. used 5×.

**For us:** retrieve 3–5× the chunk budget by relevance, then diversify *within* that pool
across questions. A node that never cleared the relevance bar can never be selected for
being novel.

### 3.2 MMR, and what λ actually trades

```
MMR = argmax [ λ·Sim₁(Dᵢ, Q) − (1−λ)· max Sim₂(Dᵢ, Dⱼ) ]
      Dᵢ∈R\S                        Dⱼ∈S
```

`λ=1` is pure relevance, `λ=0` pure diversity. Note the penalty is a **max**, not a sum —
"how similar to the single most similar thing already chosen", which is gentler and more
local than summing over everything selected.

Carbonell and Goldstein recommend starting near **λ=0.3** to explore the space, then
**λ=0.7** to focus. In their assessor study there was **no significant difference in
relevance between λ = 1, 0.7 and 0.3** — so with a properly thresholded pool, measured
relevance loss was nil. Caveat: small n, and summarisation rather than retrieval. Later
work tunes λ by cross-validation and generally does not report the chosen value, so there
is no canonical number to copy.

### 3.3 GRASSHOPPER — the direct fit, because we have a graph

Zhu et al. (NAACL 2007). An absorbing random walk over the graph:

```
P = λ·P̃ + (1−λ)·1·rᵀ          teleport interpolation
π = Pᵀπ                        stationary distribution → pick top node
                               then make it ABSORBING and repeat
v = Nᵀ·1 / (n − |G|)           expected visits before absorption
```

**Why it cannot drift**, in the authors' own words: the prior ranking `r` *"can be the
ranking by relevance scores"*, and every step teleports back into that distribution with
probability `(1−λ)`. The walk is continually re-seeded from relevance, so diversity is an
*absorption effect on top of a relevance-weighted walk* rather than a push away from it.

**And it is a better version of the original idea.** Marking a node "used" excludes exactly
that node. Absorption means a used node *proportionally suppresses its neighbourhood* —
nodes strongly connected to it get far fewer visits, while distant nodes keep theirs. On a
concept graph that is what you actually want: having asked about page replacement, you want
to move away from the whole page-replacement region, not just that one node.

Two practical notes. **λ has the opposite polarity to MMR's** — here λ=0 is pure relevance.
Do not carry a λ across methods. And the matrix inverse is only needed once: `Q` loses one
row and column per iteration and is otherwise unchanged, so Sherman–Morrison–Woodbury
applies. Reported result: ROUGE-1 0.3755 on DUC 2004, between 1st and 2nd of 34 systems,
with λ=0.5.

The authors also suggest **partial absorption** — an escape probability at absorbing
states, giving a continuum between PageRank and full absorption. That is a directly tunable
"how used is used?" dial.

---

## 4. The stronger move: explicit rather than implicit

The field separates two families:

- **Implicit** (MMR, DivRank, GRASSHOPPER): no model of what the query's aspects are.
  Diversity = dissimilarity from what is already chosen.
- **Explicit** (xQuAD, PM-2): enumerate the query's aspects, then maximise *coverage* of
  them.

Explicit is consistently stronger, and the reason matters: **it never asks "is this
different?" — it asks "which aspect is under-served, and which candidate is most *relevant*
to it?"** Relevance is the selection criterion at every step; diversity emerges from which
aspect is being served.

**The question-generation literature converged on the same answer independently.** Yadav
et al. (2024, preprint) diagnose our exact problem: neural QG suffers repeated redundant
generation *"even after utilizing several implicit techniques such as nucleus, topK
sampling, and diverse decoding"*. Their fix is explicit conditioning — on document
position, on WH-type, or on named entity:

| Approach | Type | EM | F1 |
|---|---|---|---|
| Greedy | implicit | 64.76 | 76.66 |
| Nucleus (0.95) | implicit | 64.44 | 77.15 |
| Diverse decoding | implicit | 65.21 | 77.37 |
| WH prompt | **explicit** | 67.25 | 79.60 |
| ENT prompt | **explicit** | 69.31 | 81.80 |
| **POS→WH (learned)** | **explicit** | **71.77** | **83.30** |

Explicit beats implicit by roughly **+4.1 EM / +4.5 F1**.

**For us the aspects are already there: the seed concept nodes.** Rather than "question 2
should differ from question 1", the rule becomes "question 1 covers node A, question 2
covers node B" — where both A and B cleared the relevance bar. That is explicit
diversification with the aspect enumeration handed to us by the graph, and it subsumes the
original visited-node idea while fixing its weakness.

---

## 5. When diversification actively hurts

The failure is predictable from candidate-pool quality. Dang & Croft, improvement broken
out by initial subtopic recall:

| Initial S-Recall | [0, 0.5) | [0.5, 0.75) | [0.75, 1.0] |
|---|---|---|---|
| PM-2 (α-nDCG) | +8% | +24% | **+42%** |
| xQuAD (CPR) | **−6%** | +15% | +22% |

**Below 0.5 coverage, xQuAD actively hurts.** Above 0.75 both are transformative.

Noisy aspects are worse than none: with automatically-clustered aspects, xQuAD fell *below*
the un-diversified baseline on both metrics, and the parameter-free PM-1 beat the tuned
PM-2. **If our concept-node-as-aspect mapping is noisy, explicit diversification will hurt
us** — and cov@node is currently 0.55, which is exactly the kind of signal that warrants
caution.

**Operational rule: gate diversification on pool quality rather than always applying it.**

---

## 6. Proposed design

Staged, each independently useful.

**Stage 1 — over-fetch and filter.** Retrieve 3–5× the chunk budget, apply a relevance
threshold, diversify only inside that pool. This alone provides the structural guarantee
and requires no new algorithm.

**Stage 2 — explicit node coverage.** Record which seed nodes each question drew on. For
question *n+1*, prefer candidates whose seed nodes are not yet covered, **selecting by
relevance within the uncovered set** rather than by novelty. Report
`nodes_covered / nodes_relevant` as the coverage metric — this is the "percentage of
visited nodes" idea, but computed over the relevance-filtered set so it means something.

**Stage 3 — GRASSHOPPER, if stage 2 proves insufficient.** Relevance scores as the prior
`r`, absorption for diversity, partial absorption as the tunable dial.

**Gate:** skip diversification when the relevance-filtered pool is small or cov@node is low
for that query. Better to return two similar good questions than three where the third is
off-topic.

**Metrics to report:** pairwise token overlap between generated questions (the measure
Yadav et al. use), node coverage as defined above, and mean relevance of retrieved chunks
before and after diversification — the last being the direct check that we have not drifted.

---

## 7. Honest position

The objection was correct: pure novelty-seeking does pull in irrelevant material, at a
measured ~8% precision cost in one study, and the mechanism as originally stated has no
relevance anchor. The fix is not to tune a parameter but to change where diversity is
applied — inside a relevance-filtered pool, selecting by relevance-within-uncovered-aspect
rather than by dissimilarity.

Evidence quality is mixed and should be stated as such. MMR is foundational but its
empirical content is a two-page poster with a five-user study. GRASSHOPPER is a solid
peer-reviewed NAACL paper. The QG diversity result is an unreviewed preprint. The failure-
mode evidence from Dang & Croft is peer-reviewed and the most directly actionable.

---

## References

Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for
reordering documents and producing summaries. *SIGIR '98*, 335–336.
https://doi.org/10.1145/290941.291025

Clarke, C. L. A., et al. (2008). Novelty and diversity in information retrieval evaluation.
*SIGIR '08*. https://doi.org/10.1145/1390334.1390446

Cuconasu, F., et al. (2024). The power of noise: Redefining retrieval for RAG systems.
*SIGIR '24*. https://doi.org/10.1145/3626772.3657834

Dang, V., & Croft, W. B. (2012). Diversity by proportionality: An election-based approach
to search result diversification. *SIGIR '12*. https://doi.org/10.1145/2348283.2348296

Lin, H., & Bilmes, J. (2011). A class of submodular functions for document summarization.
*ACL-HLT 2011*. https://aclanthology.org/P11-1052/

Santos, R. L. T., Macdonald, C., & Ounis (2010). Exploiting query reformulations for web
search result diversification. *WWW '10*. https://doi.org/10.1145/1772690.1772780

Sultan, M. A., et al. (2020). On the importance of diversity in question generation for QA.
*ACL 2020*, 5651–5656. https://doi.org/10.18653/v1/2020.acl-main.500

Yadav, V., et al. (2024). *Explicit diversity conditions for effective question answer
generation with large language models* (arXiv:2406.17990) [Preprint].
https://arxiv.org/abs/2406.17990

Zhai, C., Cohen, W. W., & Lafferty, J. (2003). Beyond independent relevance: Methods and
evaluation metrics for subtopic retrieval. *SIGIR '03*.
https://doi.org/10.1145/860435.860440

Zhu, X., Goldberg, A. B., Van Gael, J., & Andrzejewski, D. (2007). Improving diversity in
ranking using absorbing random walks. *NAACL HLT 2007*, 97–104.
https://aclanthology.org/N07-1013/
