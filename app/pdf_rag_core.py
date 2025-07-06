import os
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQAWithSourcesChain

VECTOR_STORE_TYPE = "FAISS"
RETRIEVAL_SCORE_THRESHOLD = 0.7

def load_pdfs(pdf_paths):
    all_docs = []
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
                text = text.strip()
                if text:
                    all_docs.append(Document(page_content=text, metadata={"source": f"{path}-p{i+1}"}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(all_docs)


def embed_docs(chunks):
    embeddings = OpenAIEmbeddings()
    if VECTOR_STORE_TYPE == "FAISS":
        return FAISS.from_documents(chunks, embeddings)
    elif VECTOR_STORE_TYPE == "Chroma":
        return Chroma.from_documents(chunks, embeddings)
    else:
        raise ValueError("Unsupported vector store type")

def query_engine(user_query, vector_store, llm):
    # ⚡️ 1. Handle general/basic queries quickly (greetings, small talk)
    general_responses = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! How can I help you?",
        "hey": "Hey! What would you like to ask?",
        "thanks": "You're welcome!",
        "thank you": "Glad I could help!",
        "bye": "Goodbye! Have a great day!",
        "what is this": "This is a PDF-based assistant. Upload a PDF and ask questions about it.",
        "who are you": "I'm a chatbot that helps answer questions based on uploaded PDFs or general knowledge."
    }

    normalized = user_query.strip().lower()
    if normalized in general_responses:
        return general_responses[normalized], True  # ✅ fast response

    # 🔍 2. Search documents with similarity scores
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    docs_with_scores = vector_store.similarity_search_with_score(user_query, k=6)

    # Debug: print similarity scores (for development only)
    print("Similarity scores:", [score for doc, score in docs_with_scores])

    # 🧠 3. Filter by threshold (ignore poor matches)
    relevant_docs = [doc for doc, score in docs_with_scores if score < RETRIEVAL_SCORE_THRESHOLD]

    if not relevant_docs:
        return "I don't know the answer.", False  # triggers fallback_logic in Streamlit

    # 🧩 4. If good docs found, use RetrievalQAWithSourcesChain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.invoke({"question": user_query})
    return result.get("answer", "No answer returned."), True


def fallback_logic(user_query, llm):
    system_prompt = (
        "You are a helpful assistant. Answer the following user query as clearly and helpfully as possible. "
        "Try to infer intent even if the question is vague. Use general knowledge."
    )
    full_prompt = f"{system_prompt}\n\nUser: {user_query}\nAssistant:"

    response = llm.invoke(full_prompt)
    return response.content if hasattr(response, 'content') else str(response)
