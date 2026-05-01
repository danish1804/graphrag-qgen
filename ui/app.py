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
    st.info("Question generation will appear here.")

elif page == "Admin":
    st.header("Admin")
    st.info("Index management controls will appear here.")
