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

# ---------------- THEME (CHATGPT STYLE) ----------------
st.markdown("""
<style>

/* Background */
html, body, [class*="css"]  {
    background-color: #0d0d0d;
    color: white;
}

/* Center container */
.main {
    max-width: 850px;
    margin: auto;
    padding-top: 40px;
}
/* Placeholder text (Type your query here...) */
.stChatInput textarea::placeholder {
    color: #9fe3d3 !important;   /* soft aqua highlight */
    opacity: 1 !important;
    font-weight: 500;
}

/* When user clicks → placeholder fades nicely */
.stChatInput textarea:focus::placeholder {
    color: #5c5f62 !important;
}


/* Title */
.title {
    text-align:center;
    font-size: 40px;
    font-weight:700;
    margin-bottom:5px;
}

.subtitle {
    text-align:center;
    color:#9ca3af;
    margin-bottom:40px;
}

/* Chat area */
.chat-container {
    display:flex;
    flex-direction:column;
    gap:18px;
}

/* Assistant message (white box) */
.bot {
    background:#ffffff;
    color:#000000;
    padding:16px 18px;
    border-radius:14px;
    max-width:75%;
    width:fit-content;
    box-shadow:0 2px 8px rgba(0,0,0,0.25);
}

/* User message (gray box) */
.user {
    background:#2a2a2a;
    color:#ffffff;
    padding:16px 18px;
    border-radius:14px;
    max-width:75%;
    width:fit-content;
    margin-left:auto;
}

/* Input box */
.stChatInput textarea {
    background:#202123 !important;     /* ChatGPT graphite */
    border:1px solid #3a3b3c !important;
    color:#ffffff !important;
    border-radius:14px !important;
    padding:14px !important;
}

/* Focus glow */
.stChatInput textarea:focus {
    border:1px solid #10a37f !important;
    box-shadow:0 0 0 1px #10a37f55;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main">', unsafe_allow_html=True)
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
You are a company HR policy assistant.

Rules:
- Respond to greetings naturally
- Answer ONLY policy related questions from context
- If not found → say: "Not mentioned in company policy."
- Be clear and professional

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

# ---------------- CHAT DISPLAY ----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot">{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
query = st.chat_input("Message Policy Assistant...")

if query:
    st.session_state.messages.append({"role":"user","content":query})

    with st.spinner("Thinking..."):
        reply = rag_chain.invoke(query)

    st.session_state.messages.append({"role":"assistant","content":reply})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


