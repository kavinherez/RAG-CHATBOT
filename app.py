import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

st.set_page_config(page_title="Company AI Assistant", layout="wide")

# ---------------- STYLE ----------------

st.markdown("""

<style>
.block-container {max-width: 900px; padding-top: 2rem;}
.chat-user {background:#1f6feb;padding:12px;border-radius:12px;color:white;margin:8px 0;}
.chat-bot {background:#262730;padding:12px;border-radius:12px;color:white;margin:8px 0;}
</style>

""", unsafe_allow_html=True)

# ---------------- API KEY ----------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ---------------- SIDEBAR ----------------

st.sidebar.title("🏢 HR Assistant")
st.sidebar.write("Upload company policy once")
uploaded_file = st.sidebar.file_uploader("Upload Policy PDF", type="pdf")

# ---------------- SESSION ----------------

if "messages" not in st.session_state:
st.session_state.messages = []

# ---------------- LOAD DOCUMENT ----------------

if uploaded_file and "vector" not in st.session_state:
with open("temp.pdf", "wb") as f:
f.write(uploaded_file.read())

```
loader = PyPDFLoader("temp.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, embeddings)

st.session_state.vector = vector_db
st.sidebar.success("Policy ready! Ask questions.")
```

# ---------------- CHAT DISPLAY ----------------

for msg in st.session_state.messages:
if msg["role"] == "user":
st.markdown(f"<div class='chat-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
else:
st.markdown(f"<div class='chat-bot'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

# ---------------- CHAT INPUT ----------------

question = st.chat_input("Ask HR anything...")

if question and "vector" in st.session_state:
st.session_state.messages.append({"role":"user","content":question})

```
retriever = st.session_state.vector.as_retriever()
docs = retriever.invoke(question)
context = "\n".join([d.page_content for d in docs])

llm = ChatGroq(model_name="llama-3.1-8b-instant")

prompt = f"""
```

You are an HR assistant. Answer ONLY from company policy.

Policy:
{context}

Question:
{question}
"""

```
answer = llm.invoke(prompt).content

st.session_state.messages.append({"role":"assistant","content":answer})
st.rerun()
```
