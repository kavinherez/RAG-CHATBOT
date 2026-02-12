import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Company Policy Assistant", layout="wide")

# ================= LOAD API KEY =================
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ================= STYLING =================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(120deg,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617,#020617);
}

/* HEADER */
.main-title {
    text-align:center;
    font-size:52px;
    font-weight:800;
    background:linear-gradient(90deg,#22d3ee,#34d399);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-top:20px;
}

.subtitle {
    text-align:center;
    color:#94a3b8;
    font-size:18px;
    margin-bottom:40px;
}

/* CHAT BUBBLES */
.user-bubble {
    background:#065f46;
    color:white;
    padding:14px 18px;
    border-radius:18px 18px 4px 18px;
    max-width:60%;
    margin-left:auto;
    margin-top:8px;
    box-shadow:0 8px 25px rgba(0,0,0,0.35);
}

.bot-bubble {
    background:#1e293b;
    color:white;
    padding:14px 18px;
    border-radius:18px 18px 18px 4px;
    max-width:60%;
    margin-right:auto;
    margin-top:8px;
    box-shadow:0 8px 25px rgba(0,0,0,0.35);
}

/* CHAT INPUT PREMIUM */
[data-testid="stChatInput"] {
    position:fixed;
    bottom:18px;
    left:50%;
    transform:translateX(-50%);
    width:70%;
}

[data-testid="stChatInput"] textarea {
    background:rgba(15,23,42,0.75) !important;
    backdrop-filter:blur(14px);
    border-radius:20px !important;
    padding:14px !important;
    color:white !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    box-shadow:0 10px 40px rgba(0,0,0,0.5);
}

[data-testid="stChatInput"] button {
    background:linear-gradient(135deg,#22d3ee,#34d399) !important;
    border-radius:14px !important;
    color:#020617 !important;
    border:none !important;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🏢 Company Policy Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your intelligent workplace knowledge companion</div>', unsafe_allow_html=True)

# ================= LOAD DOCUMENT ONCE =================
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

    prompt = ChatPromptTemplate.from_template("""
You are an internal HR assistant.
Answer ONLY from company policy.
If not found say: "Not mentioned in company policy."

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

rag_chain = load_rag()

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hello 👋 Welcome back to the office. How can I assist you today?"}
    ]

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# ================= CHAT INPUT =================
prompt = st.chat_input("Message Company Policy Assistant...")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(prompt)

    st.session_state.messages.append({"role":"assistant","content":response})
    st.rerun()
