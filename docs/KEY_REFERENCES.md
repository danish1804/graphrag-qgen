# Key References — Evaluation Methodology

**Project:** P000233DS (EdGenAI) · **Compiled by:** Mohammed Danish Alam · **19 August 2026**

A curated reading list behind the evaluation decisions in
[`EVALUATION_ROADMAP.md`](EVALUATION_ROADMAP.md). Shared so the methodology choices are
reviewable rather than assertions.

**Inclusion rule: peer-reviewed venues only.** Several relevant 2025–26 arXiv preprints were
consulted but are deliberately excluded here — they are listed in the roadmap and flagged as
preprints. Everything below has been through review at a recognised venue.

---

## If you read only three

| Paper | Venue | Why it matters to us |
|---|---|---|
| **Zheng et al. (2023)** — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | NeurIPS 2023 (D&B) | The foundational LLM-as-judge paper. Establishes the method, its agreement rates with humans, and its biases. Everything we do on evaluation sits on top of this |
| **Thakur et al. (2021)** — BEIR: A heterogeneous benchmark for zero-shot IR | NeurIPS 2021 (D&B) | The standard for retrieval evaluation. Section 3.3 explains exactly when nDCG is the right metric and when it is not, which settles our NDCG question |
| **Getahun et al. (2024)** — Experts' prediction of item difficulty | BMC Medical Education | 200 MCQs, 815 examinees, expert raters. Expert-predicted difficulty explained only **R² = 0.33** of actual difficulty. Directly relevant to how we frame our difficulty labelling |

---

## 1. Retrieval evaluation

**Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).** BEIR: A
heterogeneous benchmark for zero-shot evaluation of information retrieval models.
*NeurIPS 2021, Datasets and Benchmarks Track.* https://arxiv.org/abs/2104.08663

> Two things we use. (a) §3.3 states the reason for standardising on nDCG@10: MRR and MAP
> *"fail to evaluate tasks with graded relevance judgements."* Our judgements are binary, so
> that justification does not transfer. (b) The Hole@10 analysis shows relevance pools built
> from one retriever systematically under-score a different retriever (DPR 0.332 → 0.445
> after re-judging). Since we changed our retriever, our gold set has to be re-pooled.

**Cuconasu, F., et al. (2024).** The power of noise: Redefining retrieval for RAG systems.
*SIGIR 2024.* https://doi.org/10.1145/3626772.3657834

> The best empirical study of what a RAG retriever should actually return. Two
> counter-intuitive findings: high-scoring documents that are *not* answer-bearing degrade
> LLM output, and adding random documents can improve accuracy by up to 35%. This is why we
> track context precision, not just coverage.

**Liu, N. F., et al. (2023).** Lost in the middle: How language models use long contexts.
*TACL.* https://arxiv.org/abs/2307.03172

> Performance is highest when relevant information sits at the beginning or end of the
> context and degrades in the middle. Combined with the paper above, this means utility by
> position is U-shaped, whereas NDCG assumes it decays logarithmically — the metric's user
> model does not match how we consume chunks.

**Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024).** ARES: An automated
evaluation framework for RAG systems. *NAACL 2024.*
https://aclanthology.org/2024.naacl-long.20.pdf

> The only effect-size-calibrated guidance on evaluation-set size I found: below roughly
> **100–150 labelled datapoints** the framework cannot distinguish RAG systems that differ by
> 2.5 accuracy points. Also introduces confidence intervals for evaluation scores via
> prediction-powered inference, which is worth adopting later.

**Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024).** RAGAS: Automated
evaluation of retrieval augmented generation. *EACL 2024 (System Demonstrations).*
https://aclanthology.org/2024.eacl-demo.16.pdf

> Defines faithfulness, answer relevance and context relevance, all reference-free. Useful
> for the metrics we can compute without labels. Caveat worth knowing: ARES measured RAGAS
> context relevance accuracy at 15–36%, and the RAGAS library has since removed it from the
> core RAG metric set.

---

## 2. LLM-as-judge: bias and validation

**Zheng, L., et al. (2023).** Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.
*NeurIPS 2023, Datasets and Benchmarks Track.* https://arxiv.org/abs/2306.05685

> The canonical reference. Reports GPT-4 agreeing with humans at 85% versus human-human
> agreement of 81%. Worth reading §3.3 carefully: the authors explicitly state their study
> *"cannot determine whether the models exhibit a self-enhancement bias"* — the widely quoted
> "10% self-preference" figure is more tentative than its reputation. Also shows
> reference-guided grading cutting judge failures on maths from 14/20 to 3/20.

**Panickssery, A., Bowman, S. R., & Feng, S. (2024).** LLM evaluators recognize and favor
their own generations. *NeurIPS 2024.* https://arxiv.org/abs/2404.13076

> The strongest causal evidence for self-preference. GPT-4 recognises its own output at
> 73.5%, and self-recognition confidence correlates with self-preference (Kendall τ = 0.41,
> rising to 0.74 after fine-tuning). This is the citation for why we decoupled the judge from
> the generator.

**Wang, P., et al. (2024).** Large language models are not fair evaluators. *ACL 2024.*
https://aclanthology.org/2024.acl-long.511/

> Quantifies position bias: verdicts flip on swapping candidate order 46.3% of the time for
> GPT-4 and 82.5% for ChatGPT. Provides two practical mitigations (multiple evidence
> calibration, balanced position calibration) that we should apply to any pairwise judging.

**Thakur, A. S., Choudhary, K., Ramayapally, V. S., Vaidyanathan, S., & Hupkes, D. (2025).**
Judging the judges: Evaluating alignment and vulnerabilities in LLMs-as-judges.
*GEM 2025 (ACL workshop).* https://aclanthology.org/2025.gem-1.33/

> The most practically important finding for our reporting: **raw percent agreement is
> close to useless as a validation statistic.** A judge above 90% percent agreement can still
> misestimate system scores by more than 10 points, whereas Scott's π above 80 keeps them
> within about 5. Their recommendation is to report both. Also finds no judge reaching
> human-human parity.

**Calderon, N., Reichart, R., & Dror, R. (2025).** The alternative annotator test for
LLM-as-a-judge. *ACL 2025 (Long Papers).* https://aclanthology.org/2025.acl-long.782/

> A proper statistical procedure for the question "can this judge replace our human
> raters?", rather than a threshold rule of thumb. Leave-one-annotator-out, hypothesis test
> with an FDR correction, decision rule ω ≥ 0.5. Requires ≥3 annotators and ≥30 instances,
> which our labelling exercise will satisfy. Code: https://github.com/nitaytech/AltTest

**Bavaresco, A., et al. (2025).** LLMs instead of human judges? A large scale empirical study
across 20 NLP evaluation tasks. *ACL 2025 (Short Papers).*
https://aclanthology.org/2025.acl-short.20/

> 20 datasets, 70,000+ instances, 11 models. Agreement with humans varies enormously by task
> (κ/ρ from −0.24 to 0.84), and chain-of-thought prompting does *not* consistently improve it.
> Conclusion we should follow: validate the judge against task-specific human judgements
> before deploying it, rather than assuming transfer.

---

## 3. Question generation and assessment quality

**Fu, W., Wei, B., Hu, J., Cai, Z., & Liu, J. (2024).** QGEval: Benchmarking
multi-dimensional evaluation for question generation. *EMNLP 2024.*
https://aclanthology.org/2024.emnlp-main.658/

> Standardises seven evaluation dimensions: fluency, clarity, conciseness, relevance,
> consistency, answerability, answer consistency. This is the citable basis for our human
> evaluation rubric. Also reports that existing automatic metrics fail to align with human
> judgement across all seven.

**Mohammadshahi, A., et al. (2023).** RQUGE: Reference-free metric for evaluating question
generation by answering the question. *Findings of ACL 2023.*
https://aclanthology.org/2023.findings-acl.428/

> The method behind the AI-solvability check Samar asked for. An independent QA model answers
> the generated question and a span scorer compares it to the gold answer. No training
> required, correlates better with humans than BLEU/ROUGE, and is robust to adversarial
> corruptions that fool n-gram metrics.

**Kurdi, G., Leo, J., Parsia, B., Sattler, U., & Al-Emari, S. (2020).** A systematic review of
automatic question generation for educational purposes. *International Journal of Artificial
Intelligence in Education, 30*(1), 121–204.
https://link.springer.com/article/10.1007/s40593-019-00186-y

> The standard survey of the field, covering 93 papers. Useful context: only 14 of those 93
> addressed question difficulty at all, so difficulty control is genuinely under-served rather
> than something we have overlooked.

**Getahun, D. A., et al. (2024).** Experts' prediction of item difficulty of multiple-choice
questions in the Ethiopian Undergraduate Medicine Licensure Examination. *BMC Medical
Education, 24*, 1049. https://link.springer.com/article/10.1186/s12909-024-06012-x

> The most consequential paper for our labelling exercise. Six to seven expert physicians
> rated 200 MCQs later sat by 815 examinees. They agreed **with each other** at r = 0.73–0.84
> but with **actual** difficulty at only r = 0.46–0.55, explaining R² = 0.33 overall and as
> little as 0.001 in some domains. The implication: high inter-rater agreement on difficulty
> is not evidence the labels are right, and we should report our difficulty labels as
> *intended* difficulty.

**Yaneva, V., et al. (2024).** Findings from the first shared task on automated prediction of
difficulty and response time for multiple-choice questions. *BEA 2024 (NAACL workshop).*
https://aclanthology.org/2024.bea-1.39/

> Run by the National Board of Medical Examiners on 667 retired USMLE items with 17
> competing teams. The best system barely beat a predict-the-mean baseline. This is the
> honest state of the art for predicting difficulty from question text, and the reason we
> should treat difficulty as a limitation rather than a deliverable.

**Moore, S., Nguyen, H. A., Chen, T., & Stamper, J. (2023).** Assessing the quality of
multiple-choice questions using GPT-4 and rule-based methods. *EC-TEL 2023.*
https://arxiv.org/abs/2307.08161

> On 200 questions across four subjects, a rule-based method detected 91% of item-writing
> flaws versus GPT-4's 79%. The lesson we are adopting: use deterministic rules for
> structural quality checks and reserve the LLM for semantic judgement. The same group's
> SAQUET toolkit (AIED 2024, https://arxiv.org/abs/2405.20529) is open source and validated
> across five subject areas.

**Haladyna, T. M., Downing, S. M., & Rodriguez, M. C. (2002).** A review of multiple-choice
item-writing guidelines for classroom assessment. *Applied Measurement in Education, 15*(3),
309–333. https://eric.ed.gov/?id=EJ660246

> The standard 31-guideline taxonomy in educational measurement, validated against 27
> textbooks and 27 studies. The basis for any automated item-quality checking we build.

**Downing, S. M. (2005).** The effects of violating standard item writing principles on tests
and students. *Advances in Health Sciences Education, 10*(2), 133–143.
https://pubmed.ncbi.nlm.nih.gov/16078098/

> Why item flaws matter beyond tidiness: across four medical exams, 36–65% of items were
> flawed, flawed items ran 0–15 percentage points harder than sound items measuring the same
> content, and pass rates differed by a median of 3.5 points. Flaws distort difficulty, which
> would corrupt any calibration built on top of them.

---

## 4. Human evaluation protocol

**Howcroft, D. M., et al. (2020).** Twenty years of confusion in human evaluation: NLG needs
evaluation sheets and standardised definitions. *INLG 2020.*
https://aclanthology.org/2020.inlg-1.23/

> A review of 165 NLG papers found 204 distinct quality dimensions collapsing to 71 criteria,
> with "fluency" alone resolving into 15 different definitions. The practical takeaway for
> our rubric: define every criterion operationally with examples, and report item count,
> rater count, overlap, and the named agreement statistic.

**National Institute of Standards and Technology. (2000).** *TREC English relevance
judgements.* https://trec.nist.gov/data/reljudge_eng.html

> The labelling standard we are adopting for the retrieval gold set, and worth quoting
> directly: *"If you were writing a report on the subject of the topic and would use the
> information contained in the document in the report, then the document is relevant"* —
> judged binary, and relevant if **any** part of the document is relevant. That last rule is
> what keeps binary labelling fast and consistent between annotators.
