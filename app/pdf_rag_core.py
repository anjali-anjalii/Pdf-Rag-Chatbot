import os
import pdfplumber
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQAWithSourcesChain

# ----------  PROMPT ENGINEERING  ----------
PDF_ASSISTANT_SYSTEM_PROMPT = """
You are **PDF Insight‑Bot**, a helpful AI that answers questions
about one or more PDF documents provided as *context* below.

Your behaviour rules
====================
1. **Greeting detection** – If the user message is a pure greeting /
   gratitude / farewell (e.g. “hi”, “hello”, “thanks”), ignore the PDFs
   and reply with an appropriate short social answer.  
   *Do not* add extra details.

2. **Use‑the‑PDF‑first principle** – If the question is *not* a pure
   greeting, read the ***context*** chunks carefully and build the best
   possible answer **only from that information**.

3. **Structured understanding**  
   • recognise and reason over tables, equations, code blocks, e‑mail
     addresses, phone numbers, hyperlinks and dates that may appear
     verbatim inside the context.  
   • When relevant, include them verbatim in your answer
     (preserve formatting for code / tables).

4. **Confidence filter** – If, after careful reading, the context
   clearly *does not* contain what the user wants, respond exactly with:
   > I don’t know the answer from the provided PDFs.  
   (No extra text.)  This single line is caught by Streamlit and the
   fallback flow will trigger.

5. **Resume specialisation** – When the PDFs look like a résumé / CV  
   (they contain keywords such as “Education”, “Work Experience”,
   “Skills”, or have fewer than 10 pages and many bullet points):  
   • be ready to extract contact info, education, work history, skills.  
   • be ready to generate 5‑10 technical / interview questions relevant
     to the role mentioned in the résumé; if no role is explicit, infer
     the probable role from the résumé.

6. **Research‑paper specialisation** – For papers that contain sections
   like “Abstract”, “Methodology”, maths equations or citations:  
   • summarise theory, explain equations, interpret tables/figures.  
   • answer advanced follow‑ups about results or methods.

7. **Planner specialisation** – If the PDF contains calendar‑like
   words (“Monday”, “13:00” etc.) or a grid of days:  
   • treat it as a planner and answer queries about tasks, timings or
     probable scheduling.

8. **Formatting**  
   • Use short paragraphs, bullet‑lists or tables only when they make
     the answer clearer.  
   • Preserve code fences (```python … ```), markdown tables and
     equation LaTeX exactly as seen in context.

Now answer the user.

Context
-------
{context}

User question
-------------
{question}
"""


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
    relevant_docs = [doc for doc, score in docs_with_scores
                 if score > RETRIEVAL_SCORE_THRESHOLD]  # keep strong hits

    if not relevant_docs:
        return "I don't know the answer.", False  # triggers fallback_logic in Streamlit

    # 🧩 4. If good docs found, use RetrievalQAWithSourcesChain
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PDF_ASSISTANT_SYSTEM_PROMPT,
    )

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt},
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
