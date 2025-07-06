import os
import streamlit as st
import config
from pdf_rag_core import load_pdfs, embed_docs, query_engine, fallback_logic
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

st.set_page_config(page_title="Chat with PDFs")

# Header with reset button aligned to the right
header_col1, header_col2 = st.columns([10, 1])
with header_col1:
    st.markdown("<h1 style='margin-bottom: 0;'>PDF-Based RAG System</h1>", unsafe_allow_html=True)
with header_col2:
    if st.button("🔄", help="Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.awaiting_fallback = False
        st.session_state.last_prompt = ""
        st.session_state.vector_store = None
        st.rerun()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "awaiting_fallback" not in st.session_state:
    st.session_state.awaiting_fallback = False
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI()
if "generating" not in st.session_state:
    st.session_state.generating = False

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Loading and embedding PDFs..."):
        file_paths = [f.name for f in uploaded_files]
        for file in uploaded_files:
            with open(file.name, "wb") as out_file:
                out_file.write(file.read())
        chunks = load_pdfs(file_paths)
        if not chunks:
            st.error("No text found in the uploaded PDFs.")
            st.stop()
        st.session_state.vector_store = embed_docs(chunks)

# Show chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Handle fallback logic immediately
if st.session_state.awaiting_fallback:
    st.chat_message("assistant").write(
        "I couldn't find enough information in the documents. Would you like me to answer based on my general knowledge instead?"
    )
    with st.form("fallback_form"):
        choice = st.radio("Choose an option:", ["Yes", "No"], key="fallback_choice")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if choice == "Yes":
            with st.chat_message("assistant"):
                with st.spinner("Using general knowledge..."):
                    fallback_answer = fallback_logic(st.session_state.last_prompt, st.session_state.llm)
                    st.markdown(fallback_answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": fallback_answer})
        else:
            with st.chat_message("assistant"):
                st.markdown("Okay.")
                st.session_state.chat_history.append({"role": "assistant", "content": "Okay."})

        st.session_state.awaiting_fallback = False
        st.session_state.last_prompt = ""
        st.rerun()

# Chat input
if not st.session_state.awaiting_fallback:
    prompt = st.chat_input("Ask a question about the PDFs")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.last_prompt = prompt

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stop_button = st.empty()
            st.session_state.generating = True
            with st.spinner("Searching for answers..."):
                stop = stop_button.button("🛑 Stop Generating")
                if stop:
                    st.session_state.generating = False
                    st.rerun()

                if st.session_state.vector_store:
                    result = query_engine(prompt, st.session_state.vector_store, st.session_state.llm)
                    if isinstance(result, tuple):
                        answer, found_in_pdf = result
                    else:
                        answer, found_in_pdf = result, False

                    if not found_in_pdf:
                        st.session_state.awaiting_fallback = True
                        st.session_state.generating = False
                        st.rerun()

                    if st.session_state.generating:
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.session_state.generating = False
