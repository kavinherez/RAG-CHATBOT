import streamlit as st
import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ---------------- PAGE ----------------
st.set_page_config(page_title="HR WhatsApp Assistant", layout="wide")

# ---------------- WHATSAPP STYLE ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.imgur.com/3ZQ3Z6R.png");
    background-size: cover;
}

.header {
    background:#202c33;
    padding:14px;
    color:white;
    font-size:20px;
    border-bottom:1px solid #2a3942;
    position:sticky;
    top:0;
}

.user {
    background:#005c4b;
    color:white;
    padding:10px 14px;
    border-radius:12px 12px 0px 12px;
    margin:6px 0;
    max-width:60%;
    margin-left:auto;
    width:fit-content;
}

.bot {
    background:#202c33;
    color:white;
    padding:10px 14px;
    border-radius:12px 12px 12px 0px;
    margin:6px 0;
    max-width:60%;
    width:fit-content;
}

.time {
    font-size:10px;
    opacity:0.6;
    text-align:right;
}

.tick {
    font-size:11px;
    opacity:0.6;
    float:right;
    margin-left:6px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- API KEY ----------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ---------------- HEADER ----------------
st.markdown("<div class='header'>🟢 HR Helpdesk (online)</div>", unsafe_allow_html=True)

# ---------------- MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- LOAD VECTOR DB ----------------
@st.cache_resource
def load_vector():
    loader = PyPDFLoader("manual.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings)

    return db

retriever = load_vector().as_retriever()

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    time = datetime.now().strftime("%H:%M")

    if msg["role"] == "user":
        st.markdown(f"<div class='user'>{msg['content']}<div class='time'>{time}<span class='tick'>✔✔</span></div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>{msg['content']}<div class='time'>{time}</div></div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
question = st.chat_input("Type a message")

if question:
    st.session_state.messages.append({"role":"user","content":question})

    with st.spinner("HR typing..."):
        docs = retriever.invoke(question)
        context = "\n".join([d.page_content for d in docs])

        llm = ChatGroq(model_name="llama-3.1-8b-instant")

        prompt = f"""
You are an HR assistant.
Answer ONLY using the company policy below.
If not found, say: Not mentioned in policy.

Policy:
{context}

Question:
{question}
"""

        answer = llm.invoke(prompt).content

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()
