import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Policy Assistant", page_icon="🤖", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    background: linear-gradient(180deg,#0b1220,#020617);
    color: white;
}

/* Center container */
.main-container {
    max-width: 900px;
    margin: auto;
    padding-top: 40px;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg,#22d3ee,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.subtitle {
    text-align:center;
    color:#94a3b8;
    margin-bottom:40px;
}

/* Chat cards */
.user-card {
    background: #065f46;
    padding:14px 18px;
    border-radius:18px;
    width: fit-content;
    max-width:75%;
    margin-left:auto;
    margin-bottom:18px;
}

.bot-card {
    background:#1e293b;
    padding:14px 18px;
    border-radius:18px;
    width: fit-content;
    max-width:75%;
    margin-right:auto;
    margin-bottom:18px;
}

/* Input box */
.stTextInput>div>div>input {
    background:#0f172a;
    color:white;
    border-radius:12px;
    border:1px solid #334155;
    padding:14px;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="title">Policy Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about company rules & benefits</div>', unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hello 👋 Ask me anything about company policies."}
    ]

# ---------------- LOAD RAG ----------------
@st.cache_resource
def load_rag():
    loader = PyPDFLoader("manual.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding)
    retriever = vectordb.as_retriever()

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
You are a helpful HR assistant.

Rules:
- If greeting → respond naturally
- If question outside policy → say politely you only answer company policy questions
- If answer exists → answer clearly and shortly
- Do NOT say 'based on context'

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = load_rag()

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-card">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-card">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
query = st.chat_input("Ask a policy question...")

if query:
    st.session_state.messages.append({"role":"user","content":query})

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(query)

    st.session_state.messages.append({"role":"assistant","content":response})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
