import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Company Policy Assistant", layout="wide")

# ================= HIDE STREAMLIT DEFAULT UI =================
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}
[data-testid="stDecoration"] {display:none;}
[data-testid="stStatusWidget"] {display:none;}

.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    padding: 10px 0 20px 0;
}

.user-bubble {
    background: #075e54;
    padding: 12px 18px;
    border-radius: 18px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
}

.bot-bubble {
    background: #1f2937;
    padding: 12px 18px;
    border-radius: 18px;
    margin: 8px 0;
    width: fit-content;
    max-width: 70%;
}

.stTextInput input {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="main-title">🏢 Company Policy Assistant</div>', unsafe_allow_html=True)
st.caption("Ask anything about company rules & benefits")

# ================= LOAD GROQ KEY =================
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ================= LOAD PDF ONLY ONCE =================
@st.cache_resource
def load_rag():
    loader = PyPDFLoader("manual.pdf")  # keep pdf in repo root
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embedding_model)
    retriever = vector_db.as_retriever()

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
You are an HR Company Policy Assistant.

STRICT RULES:
1) If greeting → respond politely
2) If unrelated to company policy → say you only answer policy questions
3) If policy question → answer ONLY using the context
4) If not found → say "This is not mentioned in the company policy"
5) Never invent answers

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

# ================= SESSION CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# ================= INPUT BOX =================
user_input = st.chat_input("Type a message")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f'<div class="bot-bubble">{response}</div>', unsafe_allow_html=True)
