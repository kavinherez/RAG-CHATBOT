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
st.set_page_config(page_title="Policy Assistant", page_icon="🤖", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>

html, body, [class*="css"] {
    background: linear-gradient(180deg,#0f172a,#020617);
    color: white;
}

/* center container */
.main {
    max-width: 880px;
    margin: auto;
    padding-top: 40px;
}

/* title */
.title {
    font-size: 48px;
    font-weight: 800;
    text-align:center;
    background: linear-gradient(90deg,#22d3ee,#34d399);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle {
    text-align:center;
    color:#94a3b8;
    margin-bottom:35px;
}

/* welcome card */
.welcome {
    background: rgba(30,41,59,0.6);
    border:1px solid rgba(148,163,184,0.2);
    backdrop-filter: blur(10px);
    padding:22px;
    border-radius:18px;
    margin-bottom:30px;
}

/* suggestion buttons */
.chips {
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    margin-top:15px;
}

.chip {
    padding:8px 14px;
    background:#0f172a;
    border:1px solid #334155;
    border-radius:999px;
    font-size:14px;
    cursor:pointer;
}

/* chat bubbles */
.user {
    background:#065f46;
    padding:14px 18px;
    border-radius:16px 16px 4px 16px;
    margin:12px 0;
    width:fit-content;
    max-width:75%;
    margin-left:auto;
}

.bot {
    background:#1e293b;
    padding:14px 18px;
    border-radius:16px 16px 16px 4px;
    margin:12px 0;
    width:fit-content;
    max-width:75%;
}

/* input */
.stChatInput textarea {
    background:#020617 !important;
    border:1px solid #334155 !important;
    border-radius:14px !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">Policy Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your AI workplace knowledge companion</div>', unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- WELCOME PANEL ----------------
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome">
    👋 <b>Welcome!</b><br><br>
    I can help you understand company policies, HR rules, and employee benefits instantly.
    <br><br>
    Try asking:
    <div class="chips">
        <div class="chip">Maternity leave policy</div>
        <div class="chip">Work from home rules</div>
        <div class="chip">Notice period</div>
        <div class="chip">Leave balance</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

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
You are an HR assistant.

Rules:
- Answer greetings naturally
- Only answer company policy related questions
- If outside scope say politely you handle company policies only
- Keep answers clear and concise

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
        st.markdown(f'<div class="user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
query = st.chat_input("Ask a policy question...")

if query:
    st.session_state.messages.append({"role":"user","content":query})

    with st.spinner("Thinking..."):
        reply = rag_chain.invoke(query)

    st.session_state.messages.append({"role":"assistant","content":reply})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
