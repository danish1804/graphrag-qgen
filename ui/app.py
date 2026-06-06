import streamlit as st
import requests
from api_client import ApiClient, DEFAULT_API_BASE_URL
from config_loader import get_difficulty_retrieval, get_standard_rag_top_k

# Bloom levels from config.yml question_types (for demo labels)
BLOOM_BY_QUESTION_TYPE = {
    "recall": (1, "Remember — define, list, state, identify"),
    "conceptual": (2, "Understand — explain, describe, compare"),
    "application": (3, "Apply/Analyse — design, analyse, evaluate"),
    "case_study": (4, "Evaluate/Create — scenario then applied question"),
}

DEMO_OBJECTIVES = {
    "Case study (OS)": (
        "Apply process synchronisation concepts to prevent deadlock "
        "when multiple processes share resources"
    ),
    "Easy (database)": (
        "Explain how SQL joins combine rows from multiple tables"
    ),
    "Hard multi-domain": (
        "Compare transaction isolation in database systems with "
        "process synchronisation in operating systems"
    ),
    "Deep learning": (
        "Explain how convolutional neural networks use parameter sharing"
    ),
}


def _chunk_expander_label(name: str, chunk: dict) -> str:
    chunk_type = chunk.get("type", "chunk")
    domain = chunk.get("domain", "")
    source = chunk.get("source", "unknown")
    parts = [name, chunk_type]
    if domain:
        parts.append(domain)
    parts.append(str(source)[:40])
    return " · ".join(parts)


def _render_routing_banner(metadata: dict) -> None:
    st.subheader("Routing & retrieval")
    cols = st.columns(4)

    retrieval = metadata.get("retrieval_method", "—")
    cols[0].metric("Method", retrieval)

    routed = metadata.get("domain_routed_to") or metadata.get("domain", "—")
    cols[1].metric("Domain routed to", routed)

    domains_used = metadata.get("domains_used")
    if domains_used:
        cols[2].metric("Domains used", len(domains_used))
        cols[3].metric("Primary domain", metadata.get("domain", "—"))
        st.caption("Domains: " + ", ".join(domains_used))
        scores = metadata.get("domain_scores")
        if scores:
            st.markdown("**Domain scores (multi-domain threshold from config.yml)**")
            st.dataframe(
                [{"domain": d, "score": round(s, 3)} for d, s in scores.items()],
                hide_index=True,
                use_container_width=True,
            )
        params = metadata.get("retrieval_params")
        if params:
            st.caption(
                "Retrieval params: "
                f"top_k_nodes={params.get('top_k_nodes')}, "
                f"neighbour_depth={params.get('neighbour_depth')}, "
                f"max_chunks_per_domain={params.get('max_chunks_per_domain')} "
                f"({params.get('domains_count', 1)} domain(s))"
            )
    else:
        cols[2].metric("Domains used", 1)
        cols[3].metric("Domain", metadata.get("domain", "—"))


st.set_page_config(
    page_title="EdGenAI Question Generator",
    page_icon="ED",
    layout="wide",
)

st.title("EdGenAI Question Generator")
st.caption(
    "GraphRAG exam question generation — configurable via config.yml, "
    "with chunk transparency and LLM-as-judge scoring."
)

api_base_url = st.sidebar.text_input(
    "Backend API URL",
    value=DEFAULT_API_BASE_URL,
)

client = ApiClient(api_base_url)

with st.sidebar.expander("Demo presets (learning objectives)", expanded=False):
    for label, text in DEMO_OBJECTIVES.items():
        if st.button(label, key=f"preset_{label}", use_container_width=True):
            st.session_state["demo_objective"] = text

page = st.sidebar.radio(
    "Navigation",
    ["Generate Question", "Dashboard", "Admin"],
)

if page == "Dashboard":
    st.header("Dashboard")

    try:
        health = client.health()

        col1, col2, col3 = st.columns(3)
        col1.metric("Backend", "Online" if health.get("status") == "ok" else "Unknown")
        col2.metric("Models Loaded", "Yes" if health.get("models_loaded") else "No")
        col3.metric("API", api_base_url)

        with st.expander("Health details", expanded=False):
            st.json(health)

    except Exception as exc:
        st.error(f"Could not reach backend: {exc}")
        st.warning("Start the backend first: python main_gr.py --mode full")

    st.subheader("GraphRAG domains")
    try:
        graph_status = client.graphrag_status()
        domains = graph_status.get("domains", {})
        if domains:
            rows = [
                {
                    "domain": name,
                    "nodes": info.get("nodes", "—"),
                    "chunks": info.get("chunks", "—"),
                }
                for name, info in domains.items()
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No domains loaded.")
        with st.expander("Full GraphRAG status", expanded=False):
            st.json(graph_status)
    except Exception as exc:
        st.info(f"GraphRAG status unavailable: {exc}")

    with st.expander("Multi-document RAG status", expanded=False):
        try:
            st.json(client.multi_status())
        except Exception as exc:
            st.info(f"Unavailable: {exc}")

elif page == "Generate Question":
    st.header("Generate Question")

    retrieval_mode = st.selectbox(
        "Retrieval mode",
        [
            "GraphRAG",
            "Single-document Standard RAG",
            "Multi-document Standard RAG",
        ],
        index=0,
    )

    default_objective = st.session_state.get("demo_objective", "")
    learning_objective = st.text_area(
        "Learning objective",
        value=default_objective,
        placeholder="e.g. SQL joins and query optimisation, "
        "or: backpropagation in neural networks",
        height=120,
    )

    col1, col2 = st.columns(2)
    difficulty = col1.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)

    qt_options = list(BLOOM_BY_QUESTION_TYPE.keys())
    question_type = col2.selectbox(
        "Question type (Bloom)",
        qt_options,
        index=1,
        format_func=lambda q: (
            f"{q} — Bloom {BLOOM_BY_QUESTION_TYPE[q][0]}"
        ),
    )
    st.caption(BLOOM_BY_QUESTION_TYPE[question_type][1])

    retrieval_params = get_difficulty_retrieval(difficulty)
    if retrieval_mode == "GraphRAG":
        st.info(
            f"**Retrieval from config.yml ({difficulty}):** "
            f"`top_k_nodes={retrieval_params['top_k_nodes']}`, "
            f"`neighbour_depth={retrieval_params['neighbour_depth']}`, "
            f"`max_chunks={retrieval_params['max_chunks']}`"
        )
        top_k_nodes = retrieval_params["top_k_nodes"]
        neighbour_depth = retrieval_params["neighbour_depth"]
    else:
        top_k = get_standard_rag_top_k()
        st.info(
            f"**Retrieval from config.yml:** `top_k_results={top_k}`"
        )
    selected_domain = None

    if retrieval_mode == "GraphRAG":
        domain_options = ["Auto route"]

        try:
            graph_status = client.graphrag_status()
            domain_options.extend(list(graph_status.get("domains", {}).keys()))
        except Exception:
            pass

        domain_choice = st.selectbox("GraphRAG domain", domain_options)

        if domain_choice != "Auto route":
            selected_domain = domain_choice

        if difficulty == "hard" and domain_choice == "Auto route":
            st.info(
                "**Hard + Auto route:** multi-domain retrieval is enabled. "
                "Domains scoring at or above `domain_score_threshold` in "
                "config.yml are combined into one prompt. "
                "Select a specific domain to force single-domain retrieval. "
                "First run may take **5–15 minutes** on GPU (judge + large context)."
            )
        elif difficulty == "hard" and domain_choice != "Auto route":
            st.warning(
                f"**Hard + {domain_choice}:** using single domain only "
                "(multi-domain is skipped when a domain is specified)."
            )

    if st.button("Generate Question", type="primary"):
        if not learning_objective.strip():
            st.warning("Please enter a learning objective.")
        else:
            payload = {
                "learning_objective": learning_objective.strip(),
                "difficulty": difficulty,
                "question_type": question_type,
            }
            if retrieval_mode != "GraphRAG":
                payload["top_k"] = int(top_k)

            try:
                wait_hint = "1–5 minutes"
                if (
                    retrieval_mode == "GraphRAG"
                    and difficulty == "hard"
                    and domain_choice == "Auto route"
                ):
                    wait_hint = "5–15 minutes (multi-domain + judge)"

                with st.spinner(
                    f"Generating question, sample answer, and judge scores "
                    f"(may take {wait_hint} on GPU)..."
                ):
                    if retrieval_mode == "Single-document Standard RAG":
                        result = client.generate_single(payload)
                    elif retrieval_mode == "Multi-document Standard RAG":
                        result = client.generate_multi(payload)
                    else:
                        graph_payload = {
                            "learning_objective": learning_objective.strip(),
                            "difficulty": difficulty,
                            "question_type": question_type,
                            "top_k_nodes": top_k_nodes,
                            "neighbour_depth": neighbour_depth,
                            "domain": selected_domain,
                        }
                        result = client.generate_graphrag(graph_payload)

                st.session_state["last_result"] = result
                st.success("Question generated.")

            except requests.exceptions.ReadTimeout:
                st.error(
                    "Generation timed out waiting for the API. "
                    "Hard multi-domain runs can take 5–15 minutes — "
                    "check the API terminal; if it is still running, wait and "
                    "retry or use a single domain for a faster test."
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        metadata = result.get("metadata", {})

        if retrieval_mode == "GraphRAG" and metadata:
            _render_routing_banner(metadata)

        st.subheader("Generated Question")
        st.markdown(result.get("question", "No question returned."))

        sample_answer = metadata.get("sample_answer")
        if sample_answer:
            st.subheader("Sample Answer")
            st.markdown(sample_answer)

        judge_scores = metadata.get("judge_scores")
        if judge_scores:
            st.subheader("LLM Judge Scores")
            jcol1, jcol2, jcol3, jcol4 = st.columns(4)
            jcol1.metric(
                "Question Quality",
                f"{judge_scores.get('question_overall', 'N/A')}/5",
            )
            jcol2.metric(
                "Answer Quality",
                f"{judge_scores.get('answer_overall', 'N/A')}/5",
            )
            jcol3.metric(
                "Retrieval Quality",
                f"{judge_scores.get('retrieval_overall', 'N/A')}/5",
            )
            jcol4.metric(
                "Attempts",
                judge_scores.get("attempts", 1),
            )
            eval_id = judge_scores.get("eval_id", "")
            if eval_id:
                st.caption(f"Evaluation saved: {eval_id}")

        sources = metadata.get("sources")
        if sources:
            st.subheader("Sources Used")
            st.write(sources)

        key_concepts = metadata.get("key_concepts")
        if key_concepts:
            st.subheader("Key Concepts")
            st.write(key_concepts)

        relationships = metadata.get("concept_relationships")
        if relationships:
            st.subheader("Concept Relationships")
            st.write(relationships)

        retrieved_chunks = metadata.get("retrieved_chunks")
        if retrieved_chunks:
            st.subheader("Retrieved Chunks (prompt context)")
            st.caption(
                "Seed = direct graph match to the objective; "
                "neighbour = related concepts via graph edges."
            )
            for chunk_name, chunk in retrieved_chunks.items():
                label = _chunk_expander_label(chunk_name, chunk)
                with st.expander(label):
                    text = chunk.get("text", "")
                    if text:
                        st.markdown(text)
                    st.json(chunk)

        with st.expander("Raw API Response", expanded=False):
            st.json(result)


elif page == "Admin":
    st.header("Admin")
    st.caption(
        "Central GraphRAG database — uploads and incremental rebuilds "
        "affect all users until per-educator indexes are added."
    )

    st.subheader("Document Registry")
    if st.button("Check Unregistered Documents", key="check_registry_btn"):
        try:
            result = client.new_documents()
            st.json(result)
        except Exception as exc:
            st.error(f"Could not load registry: {exc}")

    st.divider()

    st.subheader("Upload Document To Domain")
    try:
        graph_status = client.graphrag_status()
        domains = graph_status.get("domains", {})
        domain_names = list(domains.keys())

        if domain_names:
            upload_domain = st.selectbox(
                "Upload target domain",
                domain_names,
                key="admin_upload_domain",
            )
            admin_file = st.file_uploader(
                "Upload PDF or PPTX",
                type=["pdf", "pptx"],
                key="admin_domain_upload",
            )
            if admin_file and st.button(
                "Upload To Selected Domain",
                key="upload_domain_btn",
            ):
                with st.spinner(f"Uploading to {upload_domain}..."):
                    result = client.admin_upload_document(
                        upload_domain, admin_file
                    )
                if result.get("saved_to_disk") is False and result.get(
                    "registered"
                ) is False:
                    st.warning(result.get("message", "Duplicate upload rejected."))
                elif result.get("already_indexed"):
                    st.info(result.get("message", "Already indexed."))
                else:
                    st.success(
                        "Document saved. Trigger incremental rebuild below."
                    )
                st.json(result)
        else:
            st.info("No GraphRAG domains available.")

    except Exception as exc:
        st.error(f"Could not load domains: {exc}")

    st.divider()

    st.subheader("Rebuild Domain Graph (Incremental)")
    try:
        graph_status = client.graphrag_status()
        domains = graph_status.get("domains", {})
        domain_names = list(domains.keys())

        if domain_names:
            rebuild_domain = st.selectbox(
                "Select domain to rebuild",
                domain_names,
                key="rebuild_domain_select",
            )
            with st.expander("Domain stats", expanded=False):
                st.json(domains[rebuild_domain])

            if st.button(
                "Rebuild Domain (Incremental)",
                key="rebuild_domain_btn",
            ):
                with st.spinner(
                    f"Rebuilding {rebuild_domain} incrementally..."
                ):
                    result = client.admin_rebuild_domain(rebuild_domain)
                elapsed = result.get("elapsed_seconds")
                msg = result.get("message", "Rebuild complete.")
                if elapsed is not None:
                    st.success(f"{msg} ({elapsed:.1f}s)")
                else:
                    st.success(msg)
                st.json(result)
        else:
            st.info("No GraphRAG domains found.")

    except Exception as exc:
        st.error(f"Could not load domains: {exc}")

    st.divider()

    with st.expander("Rebuild Multi-document Index (optional)", expanded=False):
        if st.button("Rebuild Multi-document Index", key="rebuild_multi_btn"):
            try:
                with st.spinner("Rebuilding combined index..."):
                    result = client.rebuild_multi_index()
                st.success("Multi-document index rebuilt.")
                st.json(result)
            except Exception as exc:
                st.error(f"Multi-document rebuild failed: {exc}")
