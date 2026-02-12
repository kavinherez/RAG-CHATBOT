import streamlit as st
import os
from datetime import datetime

# LangChain / RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= PAGE =================
st.set_page_config(page_title="Company Policy Assistant", layout="wide")

# ================= LOAD SECRET =================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ================= STYLE (WHATSAPP UI) =================
st.markdown("""
<style>
.stApp {
    background-image: url("whatsappback.png");
    background-size: cover;
}

.chat-container {
    max-width: 900px;
    margin: auto;
}

.user-msg {
    background: #075E54;
    color: white;
    padding: 12px 16px;
    border-radius: 15px 15px 0px 15px;
    margin: 8px 0;
    width: fit-content;
    margin-left: auto;
}

.bot-msg {
    background: #1f2c34;
    color: white;
    padding: 12px 16px;
    border-radius: 15px 15px 15px 0px;
    margin: 8px 0;
    width: fit-content;
}

.timestamp {
    font-size: 10px;
    opacity: 0.6;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

st.title("🏢 Company Policy Assistant")
st.caption("Ask anything about HR rules, benefits & policies")

# ================= LOAD RAG ONLY ONCE =================
@st.cache_resource
def load_rag():

    loader = PyPDFLoader("manual.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(chunks, embedding)

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
You are a professional HR assistant chatbot.

Rules:
1) If the user greets or talks casually, reply politely like a human assistant.
2) If the question is about company policy, answer ONLY from the provided context.
3) If answer not found, say: "This is not mentioned in the company policy document."

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

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= CHAT DISPLAY =================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, msg, time in st.session_state.messages:

    if role == "user":
        st.markdown(f"""
        <div class="user-msg">
            {msg}
            <div class="timestamp">{time}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="bot-msg">
            {msg}
            <div class="timestamp">{time}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ================= INPUT =================
user_input = st.chat_input("Type a message")

if user_input:

    now = datetime.now().strftime("%H:%M")

    # Save user message
    st.session_state.messages.append(("user", user_input, now))

    # Get response
    response = rag_chain.invoke(user_input)

    # Save bot message
    st.session_state.messages.append(("bot", response, now))

    st.rerun()
