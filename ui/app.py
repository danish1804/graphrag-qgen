import streamlit as st
from api_client import ApiClient, DEFAULT_API_BASE_URL

st.set_page_config(
    page_title="EdGenAI Question Generator",
    page_icon="ED",
    layout="wide",
)

st.title("EdGenAI Question Generator")
st.caption("UI for Standard RAG, Multi-document RAG, and GraphRAG workflows.")
api_base_url = st.sidebar.text_input(
    "Backend API URL",
    value=DEFAULT_API_BASE_URL,
)

client = ApiClient(api_base_url)


page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload Document", "Generate Question", "Admin"],
)

if page == "Dashboard":
    st.header("Dashboard")

    try:
        health = client.health()

        col1, col2 = st.columns(2)
        col1.metric("Backend", "Online" if health.get("status") == "ok" else "Unknown")
        col2.metric("Models Loaded", "Yes" if health.get("models_loaded") else "No")

        with st.expander("Raw Health Response", expanded=True):
            st.json(health)

    except Exception as exc:
        st.error(f"Could not reach backend: {exc}")
        st.warning("Start the backend first, for example: python main_gr.py --mode standard")
    
    st.subheader("Multi-document RAG")
    try:
        st.json(client.multi_status())
    except Exception as exc:
        st.info(f"Multi-document status unavailable: {exc}")

    st.subheader("GraphRAG")
    try:
        st.json(client.graphrag_status())
    except Exception as exc:
        st.info(f"GraphRAG status unavailable: {exc}")

elif page == "Upload Document":
    st.header("Upload Document")

    upload_mode = st.radio(
        "Upload target",
        ["Standard RAG", "GraphRAG"],
        horizontal=True,
    )

    uploaded_file = st.file_uploader(
        "Upload course material",
        type=["pdf", "pptx"],
    )

    if uploaded_file and st.button("Upload and Index", type="primary"):
        try:
            with st.spinner("Uploading and indexing document..."):
                if upload_mode == "Standard RAG":
                    result = client.upload_standard_document(uploaded_file)
                else:
                    result = client.upload_graphrag_document(uploaded_file)

            st.success("Document indexed successfully.")
            st.json(result)

        except Exception as exc:
            st.error(f"Upload failed: {exc}")

elif page == "Generate Question":
    st.header("Generate Question")

    retrieval_mode = st.selectbox(
        "Retrieval mode",
        [
            "Single-document Standard RAG",
            "Multi-document Standard RAG",
            "GraphRAG",
        ],
    )

    learning_objective = st.text_area(
        "Learning objective",
        placeholder="Explain how parameter sharing reduces learnable weights in CNNs.",
        height=120,
    )

    col1, col2, col3 = st.columns(3)
    difficulty = col1.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    question_type = col2.selectbox("Question type", ["recall", "conceptual", "application"], index=1)
    top_k = col3.number_input("Top K", min_value=1, max_value=10, value=3)
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


    if st.button("Generate Question", type="primary"):
        if not learning_objective.strip():
            st.warning("Please enter a learning objective.")
        else:
            payload = {
                "learning_objective": learning_objective.strip(),
                "difficulty": difficulty,
                "question_type": question_type,
                "top_k": int(top_k),
            }

            try:
                with st.spinner("Generating question..."):
                    if retrieval_mode == "Single-document Standard RAG":
                        result = client.generate_single(payload)
                    elif retrieval_mode == "Multi-document Standard RAG":
                        result = client.generate_multi(payload)
                    else:
                        graph_payload = {
                            "learning_objective": learning_objective.strip(),
                            "difficulty": difficulty,
                            "question_type": question_type,
                            "top_k_nodes": int(top_k),
                            "neighbour_depth": 1,
                            "domain": selected_domain,
                        }
                        result = client.generate_graphrag(graph_payload)

                st.session_state["last_result"] = result
                st.success("Question generated.")

            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        metadata = result.get("metadata", {})

        st.subheader("Generated Question")
        st.markdown(result.get("question", "No question returned."))

        if metadata:
            st.subheader("Metadata")
            st.json(metadata)

        with st.expander("Raw API Response"):
            st.json(result)




elif page == "Admin":
    st.header("Admin")
    st.info("Index management controls will appear here.")
