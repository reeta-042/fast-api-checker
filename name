import os
import uuid
import streamlit as st
from app.loaders import load_and_chunk_pdf
from app.vectorbase import store_chunks, get_vectorstore, get_bm25_retriever
from app.chain import build_llm_chain, retrieve_hybrid_docs, rerank_documents
from app.streamlit import upload_pdfs, save_uploaded_files

# Caching helpers
from streamlit.runtime.caching import cache_data, cache_resource

# 🔑 API Keys
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Page setup
st.set_page_config(page_title="📄 Chat with your PDF and prep for your exams", layout="wide")
st.title("💻 ExamAI: Chat with your Course Material")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# STEP 1: Upload PDFs
uploaded_files, submitted = upload_pdfs()


@cache_data(show_spinner=False)
def cached_chunk_pdf(file_path: str):
    return load_and_chunk_pdf(file_path)


@cache_resource
def cached_get_vectorstore(api_key, index_name, namespace):
    return get_vectorstore(api_key, index_name, namespace)


# STEP 2: Store / Load Vectorstore
if submitted and uploaded_files:
    # Create a fresh namespace for this upload
    namespace = f"session_{uuid.uuid4().hex}"
    st.session_state["current_namespace"] = namespace

    file_paths = save_uploaded_files(uploaded_files)

    all_chunks = []
    for path in file_paths:
        chunks = cached_chunk_pdf(path)
        all_chunks.extend(chunks)

    with st.spinner("📥 Ingesting and indexing your PDFs..."):
        vectorstore = store_chunks(
            all_chunks,
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            namespace=namespace
        )
        bm25 = get_bm25_retriever(all_chunks)

    st.sidebar.success(f"✅ Uploaded {len(uploaded_files)} file(s) successfully!")

else:
    # Reuse namespace if it exists
    if "current_namespace" in st.session_state:
        namespace = st.session_state["current_namespace"]
        vectorstore = cached_get_vectorstore(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            namespace=namespace
        )
    else:
        st.warning("⚠️ Please upload at least one PDF.")
        st.stop()


# STEP 3: User Query
st.subheader("...Ask Away...")
query = st.text_input("What do you want to know?")


# STEP 4: Containers
answer_container = st.empty()
followup_container = st.empty()
quiz_container = st.empty()


# STEP 5: Processing Query
if query:

    with st.spinner("🔍 Searching your course material..."):
        retrieved_docs = retrieve_hybrid_docs(query, vectorstore)

    with st.spinner("📚 Reranking..."):
        reranked_docs = rerank_documents(query, retrieved_docs)

    # ✅ Safety net: fallback if reranked_docs is empty
    if not reranked_docs and retrieved_docs:
        reranked_docs = retrieved_docs

    answer_chain, followup_chain, quiz_chain = build_llm_chain(api_key=GOOGLE_API_KEY)

    input_data = {
        "context": "\n\n".join([doc.page_content for doc in reranked_docs]),
        "question": query
    }

    with st.spinner("⌨️ Generating answer..."):
        answer = answer_chain.invoke(input_data)
        answer_container.markdown(answer)

    with st.spinner("👀 Generating follow-up..."):
        followup = followup_chain.invoke(input_data)
        followup_container.markdown(followup)

    with st.spinner("📝 Generating quiz..."):
        quiz_card = quiz_chain.invoke(input_data)

    # STEP 6: Render Quiz
    if quiz_card:
        quiz_box = quiz_container.container()
        with quiz_box:
            st.markdown("## 📝 Learn Through Quiz")
            for i, q in enumerate(quiz_card):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                for label, opt in q["options"].items():
                    if opt:  # avoid None
                        st.markdown(f"- {label}. {opt}")
                st.markdown(f"✅ **Correct Answer:** {q['answer']}")
                if q["explanation"]:
                    st.markdown(f"💡 *Why?* {q['explanation']}")
                st.markdown("---")
    else:
        st.warning("⚠️ Quiz could not be generated. Please check your prompt or context.")
