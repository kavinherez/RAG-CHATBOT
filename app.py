import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- PAGE ----------------
st.set_page_config(page_title="Policy Assistant", layout="wide")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SECURITY FILTER ----------------
def is_policy_question(question: str) -> bool:
    blocked_topics = [
        "model","architecture","training","trained","dataset",
        "who made you","openai","chatgpt","transformer",
        "ignore instructions","system prompt","joke","story",
        "weather","news","politics","president","math",
        "code","python","java","program","translate"
    ]
    q = question.lower()
    return not any(word in q for word in blocked_topics)

# ---------------- OUTPUT VALIDATOR ----------------
def validate_answer(answer: str) -> str:
    forbidden = [
        "language model","trained on","transformer",
        "architecture","neural network","openai","chatgpt"
    ]
    for w in forbidden:
        if w in answer.lower():
            return "I can only answer questions related to the company policy document."
    return answer

# ---------------- LOAD POLICY (ONCE) ----------------
@st.cache_resource
def load_rag():
    loader = PyPDFLoader("manual.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embedding_model)

    retriever = vector_db.as_retriever()

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    SYSTEM_PROMPT = """
You are an internal company HR policy assistant.

Your job is NOT to summarize policies.
Your job is to answer the employee's exact question using the policy.

Steps you must follow:
1. Understand what the employee is asking (approval, duration, eligibility, action, restriction, etc.)
2. Search the provided policy context only for information relevant to that intent
3. Answer specifically for that intent

Rules:
- If question asks "what should I do" → give steps
- If question asks "approval" → mention authorities/people
- If question asks "allowed or not" → give yes/no + condition
- Do NOT include unrelated policy details
- If policy does not mention it → say "Not mentioned in company policy."

Never hallucinate. Only use provided context.

"""

    prompt = ChatPromptTemplate.from_template("""
{system}

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x,
            "system": lambda x: SYSTEM_PROMPT
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

rag_chain = load_rag()

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;margin-top:30px;'>Policy Assistant</h1>
<p style='text-align:center;color:gray;'>Ask anything about company rules & benefits</p>
""", unsafe_allow_html=True)

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style='display:flex;justify-content:flex-end;margin:10px 0;'>
            <div style='background:#2b2b2b;color:white;padding:12px 18px;border-radius:18px;max-width:60%;'>
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='display:flex;justify-content:flex-start;margin:10px 0;'>
            <div style='background:white;color:black;padding:12px 18px;border-radius:18px;max-width:60%;box-shadow:0 2px 8px rgba(0,0,0,0.15);'>
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.chat_input("Message Policy Assistant...")

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})

    if not is_policy_question(user_input):
        reply = "I can only answer questions related to the company policy document."
    else:
        reply = rag_chain.invoke(user_input)
        reply = validate_answer(reply)

    st.session_state.messages.append({"role":"assistant","content":reply})
    st.rerun()

