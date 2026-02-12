import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= PAGE =================
st.set_page_config(page_title="Company Policy Assistant", layout="wide")

# ================= SECRET KEY =================
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ================= CSS =================
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}

.stApp {
    background: linear-gradient(135deg,#020617,#0f172a);
    color:white;
    font-family: 'Inter', sans-serif;
}

/* HEADER */
.hero { text-align:center; padding:50px 0 20px 0; }
.hero-icon { font-size:65px; margin-bottom:10px; filter: drop-shadow(0px 8px 20px rgba(0,255,200,0.4)); }
.hero-title {
    font-size:56px;
    font-weight:800;
    background: linear-gradient(90deg,#22d3ee,#34d399,#60a5fa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.hero-subtitle { color:#94a3b8; font-size:18px; }

/* CHAT */
.user-bubble {
    background:#075e54;
    padding:12px 18px;
    border-radius:18px;
    margin:10px 0;
    width:fit-content;
    max-width:70%;
    margin-left:auto;
    box-shadow:0 4px 18px rgba(0,255,170,0.15);
}

.bot-bubble {
    background:#1f2937;
    padding:12px 18px;
    border-radius:18px;
    margin:10px 0;
    width:fit-content;
    max-width:70%;
    box-shadow:0 4px 18px rgba(0,0,0,0.25);
}

/* FLOAT INPUT */
.chat-footer {
    position:fixed;
    bottom:20px;
    left:50%;
    transform:translateX(-50%);
    width:70%;
    background:rgba(15,23,42,0.75);
    backdrop-filter: blur(14px);
    border-radius:20px;
    padding:12px 16px;
    box-shadow:0 10px 40px rgba(0,0,0,0.5);
    border:1px solid rgba(255,255,255,0.08);
}

.chat-footer textarea {
    background:transparent !important;
    color:white !important;
    border:none !important;
    font-size:16px !important;
}

.chat-footer button {
    background:linear-gradient(135deg,#22d3ee,#34d399) !important;
    border:none !important;
    border-radius:14px !important;
    padding:10px 18px !important;
    font-weight:600 !important;
    color:#020617 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="hero">
<div class="hero-icon">🏢</div>
<div class="hero-title">Company Policy Assistant</div>
<div class="hero-subtitle">Your intelligent workplace knowledge companion</div>
</div>
""", unsafe_allow_html=True)

# ================= LOAD PDF (PERMANENT BACKEND) =================
@st.cache_resource
def load_rag():
    loader = PyPDFLoader("manual.pdf")  # keep pdf in repo root
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embed)
    retriever = db.as_retriever()

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
You are a company HR assistant.
Answer ONLY from company policy.

If not found in policy → say:
"Not mentioned in company policy."

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag

rag_chain = load_rag()

# ================= CHAT STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = ""

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# ================= INPUT BAR =================
st.markdown('<div class="chat-footer">', unsafe_allow_html=True)
col1, col2 = st.columns([8,1])

with col1:
    user_input = st.text_area("", placeholder="Message Company Policy Assistant...", key="input", label_visibility="collapsed")

with col2:
    send = st.button("➤")

st.markdown('</div>', unsafe_allow_html=True)

# ================= SEND MESSAGE =================
if send and user_input and user_input != st.session_state.last_user_message:
    st.session_state.last_user_message = user_input
    st.session_state.messages.append({"role":"user","content":user_input})

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_input)

    st.session_state.messages.append({"role":"assistant","content":response})
    st.rerun()
