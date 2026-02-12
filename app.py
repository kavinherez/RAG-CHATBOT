import streamlit as st
import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

st.set_page_config(page_title="HR WhatsApp Assistant", layout="wide")

# STYLE
st.markdown("""
<style>
body {background:#111b21;}
.header {background:#202c33;padding:15px;color:white;font-size:20px;}
.user {background:#005c4b;color:white;padding:10px;border-radius:12px 12px 0px 12px;margin:6px 0;width:fit-content;margin-left:auto;}
.bot {background:#202c33;color:white;padding:10px;border-radius:12px 12px 12px 0px;margin:6px 0;width:fit-content;}
.time {font-size:10px;opacity:0.6;text-align:right;}
</style>
""", unsafe_allow_html=True)

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.markdown("<div class='header'>🏢 HR Helpdesk</div>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_vector():
    loader = PyPDFLoader("manual.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)

retriever = load_vector().as_retriever()

for msg in st.session_state.messages:
    time = datetime.now().strftime("%H:%M")
    if msg["role"] == "user":
        st.markdown(f"<div class='user'>{msg['content']}<div class='time'>{time}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>{msg['content']}<div class='time'>{time}</div></div>", unsafe_allow_html=True)

question = st.chat_input("Type a message")

if question:
    st.session_state.messages.append({"role":"user","content":question})

    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = f"""
You are an HR assistant. Answer only using company policy.

Policy:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt).content

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()
