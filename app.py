import os
import time
import base64
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- PAGE ----------------
st.set_page_config(page_title="Company Policy Assistant", page_icon="🏢", layout="wide")

# ---------------- BACKGROUND IMAGE ----------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("whatsappback.png")

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{bg}");
    background-size: cover;
}}

.block-container {{
    padding-top: 1rem;
}}

.chat-bubble-user {{
    background: #005c4b;
    color: white;
    padding: 12px 18px;
    border-radius: 15px 15px 3px 15px;
    margin: 8px 0;
    max-width: 70%;
    margin-left: auto;
}}

.chat-bubble-bot {{
    background: #202c33;
    color: white;
    padding: 12px 18px;
    border-radius: 15px 15px 15px 3px;
    margin: 8px 0;
    max-width: 70%;
}}

.typing {{
    color: #aaa;
    font-style: italic;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🏢 Company Policy Assistant")
st.caption("Ask anything about company rules & benefits")

# ---------------- LOAD SECRET ----------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ---------------- LOAD & CACHE RAG ----------------
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
Answer ONLY from the provided company policy.
If answer not found, say: "Not mentioned in company policy."

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

rag_chain = load_rag()

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT ----------------
for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"<div class='chat-bubble-user'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>{msg}</div>", unsafe_allow_html=True)

# ---------------- INPUT BOX ----------------
user_input = st.chat_input("Type a message")

if user_input:

    # store user msg
    st.session_state.messages.append(("user", user_input))
    st.markdown(f"<div class='chat-bubble-user'>{user_input}</div>", unsafe_allow_html=True)

    # typing animation
    typing_placeholder = st.empty()
    typing_placeholder.markdown("<div class='typing'>Assistant is typing...</div>", unsafe_allow_html=True)

    # get response
    answer = rag_chain.invoke(user_input)

    time.sleep(0.7)
    typing_placeholder.empty()

    # store bot msg
    st.session_state.messages.append(("bot", answer))
    st.markdown(f"<div class='chat-bubble-bot'>{answer}</div>", unsafe_allow_html=True)
