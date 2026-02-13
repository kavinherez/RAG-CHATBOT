import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= CONFIG =================
st.set_page_config(page_title="Company Policy Assistant", layout="wide")
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ================= PREMIUM CSS =================
st.markdown("""
<style>

/* Hide Streamlit default input */
[data-testid="stChatInput"] { display:none; }

/* Background */
.stApp {
    background: radial-gradient(circle at top,#020617,#020617,#020617,#020617,#020617);
    font-family: Inter, sans-serif;
}

/* Header */
.main-title{
text-align:center;
font-size:54px;
font-weight:800;
background:linear-gradient(90deg,#22d3ee,#34d399);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-top:25px;
}
.subtitle{
text-align:center;
color:#94a3b8;
margin-bottom:40px;
}

/* Chat bubbles */
.user{
background:#065f46;
color:white;
padding:14px 18px;
border-radius:18px 18px 4px 18px;
max-width:60%;
margin-left:auto;
margin-top:8px;
box-shadow:0 8px 25px rgba(0,0,0,.35);
}

.bot{
background:#1e293b;
color:white;
padding:14px 18px;
border-radius:18px 18px 18px 4px;
max-width:60%;
margin-right:auto;
margin-top:8px;
box-shadow:0 8px 25px rgba(0,0,0,.35);
}

/* Floating Composer */
.chatbox{
position:fixed;
bottom:18px;
left:50%;
transform:translateX(-50%);
width:70%;
background:rgba(15,23,42,.7);
backdrop-filter:blur(18px);
border-radius:22px;
padding:12px;
display:flex;
gap:10px;
align-items:center;
box-shadow:0 10px 40px rgba(0,0,0,.5);
border:1px solid rgba(255,255,255,.08);
}

/* Textarea */
.chatbox textarea{
flex:1;
background:transparent;
border:none;
outline:none;
color:white;
font-size:16px;
resize:none;
height:26px;
max-height:120px;
}

/* Send button */
.sendbtn{
background:linear-gradient(135deg,#22d3ee,#34d399);
border:none;
color:#020617;
width:44px;
height:44px;
border-radius:50%;
font-size:20px;
cursor:pointer;
transition:.2s;
}
.sendbtn:hover{ transform:scale(1.08); }

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🏢 Company Policy Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your intelligent workplace knowledge companion</div>', unsafe_allow_html=True)

# ================= LOAD RAG =================
@st.cache_resource
def load_chain():
    loader = PyPDFLoader("manual.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
You are an HR assistant.
Answer ONLY from company policy.
If not found say: "Not mentioned in company policy."

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\\n\\n".join(d.page_content for d in docs)

    return ({"context": retriever | format_docs, "question": lambda x: x}
            | prompt | llm | StrOutputParser())

rag = load_chain()

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages=[{"role":"assistant","content":"Hello 👋 How can I help you today?"}]

# ================= CHAT DISPLAY =================
for m in st.session_state.messages:
    if m["role"]=="user":
        st.markdown(f'<div class="user">{m["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot">{m["content"]}</div>', unsafe_allow_html=True)

# ================= CUSTOM INPUT =================
# ===== CHAT INPUT FORM (FIXED VERSION) =====
# ================= PREMIUM CHAT INPUT =================
st.markdown("""
<style>
.chat-input-container{
    position: fixed;
    bottom: 18px;
    left: 50%;
    transform: translateX(-50%);
    width: min(900px, 92%);
    background: rgba(10,18,40,0.65);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 40px rgba(0,0,0,.45);
}

.chat-input textarea{
    background: transparent !important;
    color: white !important;
    border: none !important;
    font-size: 16px !important;
}

.chat-input textarea:focus{
    outline: none !important;
    box-shadow: none !important;
}

.send-btn{
    position:absolute;
    right:22px;
    bottom:22px;
    background: linear-gradient(135deg,#18d4c3,#1da1f2);
    border-radius: 50%;
    width:48px;
    height:48px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:20px;
    color:white;
    border:none;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

col1, col2 = st.columns([12,1])

with col1:
    user_input = st.text_area(
        "",
        key="live_input",
        placeholder="Message Company Policy Assistant...",
        label_visibility="collapsed",
        height=55
    )

with col2:
    send_clicked = st.button("➤", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ===== SEND LOGIC =====
if (send_clicked or (user_input and user_input.endswith("\n"))):

    clean_input = user_input.strip()

    if clean_input != "":
        st.session_state.messages.append({"role":"user","content":clean_input})

        with st.spinner("Thinking..."):
            reply = rag.invoke(clean_input)

        st.session_state.messages.append({"role":"assistant","content":reply})

        
        st.rerun()

