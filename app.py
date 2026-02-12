import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

st.set_page_config(page_title="Company Policy Assistant", layout="wide")

# API key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.title("🏢 Company Policy Assistant")
st.caption("Ask anything about company rules & benefits")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vector database once
@st.cache_resource
def load_vector_db():
    loader = PyPDFLoader("manual.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings)
    return vector_db

vector_db = load_vector_db()
retriever = vector_db.as_retriever()

# Show chat
for msg in st.session_state.messages:
    role = "🧑" if msg["role"] == "user" else "🤖"
    st.markdown(f"**{role} {msg['content']}**")

# Input
question = st.chat_input("Ask about company policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = f"""
You are a company HR assistant.
Answer ONLY from the policy below.
If not found say: Not mentioned in policy.

Policy:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt).content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
