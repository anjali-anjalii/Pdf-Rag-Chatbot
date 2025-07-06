import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQAWithSourcesChain

VECTOR_STORE_TYPE = "FAISS"
RETRIEVAL_SCORE_THRESHOLD = 0.7

def load_pdfs(pdf_paths):
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    docs_with_scores = vector_store.similarity_search_with_score(user_query, k=6)

    # Debug: print scores
    print("Similarity scores:", [score for doc, score in docs_with_scores])

    # Filter docs under threshold
    relevant_docs = [doc for doc, score in docs_with_scores if score < RETRIEVAL_SCORE_THRESHOLD]

    if not relevant_docs:
        return "I don't know the answer.", False

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.invoke({"question": user_query})
    return result.get("answer", "No answer returned."), True

def fallback_logic(user_query, llm):
    response = llm.invoke(user_query)
    return response.content if hasattr(response, 'content') else str(response)
