import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ------------------ PAGE CONFIG ------------------

st.set_page_config(page_title="Company AI Assistant", layout="wide")

# ------------------ CUSTOM CSS ------------------

st.markdown("""

<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

.chat-message {
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 10px;
    display: flex;
    gap: 12px;
}

.user {
    background-color: #1f6feb;
    color: white;
}

.bot {
    background-color: #262730;
    color: white;
}

.avatar {
    font-size: 22px;
}

.stChatInputContainer {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 900px;
}

</style>

""", unsafe_allow_html=True)

# ------------------ LOAD KEY ------------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ------------------ SIDEBAR ------------------

with st.sidebar:
st.title("🏢 HR Assistant")
st.write("Upload company policy once")

```
uploaded_file = st.file_uploader("Upload Policy PDF", type="pdf")
```

# ------------------ SESSION ------------------

if "messages" not in st.session_state:
st.session_state.messages = []

# ------------------ LOAD DOC ------------------

if uploaded_file and "vector" not in st.session_state:

```
with open("temp.pdf", "wb") as f:
    f.write(uploaded_file.read())

loader = PyPDFLoader("temp.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, embeddings)

st.session_state.vector = vector_db
st.success("Policy ready! You can ask questions.")
```

# ------------------ CHAT DISPLAY ------------------

for msg in st.session_state.messages:
role_class = "user" if msg["role"] == "user" else "bot"
avatar = "🧑" if msg["role"] == "user" else "🤖"

```
st.markdown(f"""
<div class="chat-message {role_class}">
    <div class="avatar">{avatar}</div>
    <div>{msg["content"]}</div>
</div>
""", unsafe_allow_html=True)
```

# ------------------ CHAT INPUT ------------------

prompt = st.chat_input("Ask HR anything...")

if prompt and "vector" in st.session_state:

```
st.session_state.messages.append({"role": "user", "content": prompt})

retriever = st.session_state.vector.as_retriever()
docs = retriever.invoke(prompt)
context = "\n".join([d.page_content for d in docs])

llm = ChatGroq(model_name="llama-3.1-8b-instant")

template = f"""
You are a helpful HR assistant.
Answer only using company policy.

Policy:
{context}

Question: {prompt}
"""

response = llm.invoke(template).content

st.session_state.messages.append({"role": "assistant", "content": response})
st.rerun()
```
