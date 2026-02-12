import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Company Policy Assistant", layout="wide")
st.title("🏢 Company Policy Assistant")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload Company Policy PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever()

    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
    You are an HR assistant answering employee questions.

    Use ONLY the policy context.
    If not found say: "Not available in policy document."

    Context:
    {context}

    Conversation:
    {history}

    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask about company policy")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        history_text = "\n".join([m["content"] for m in st.session_state.messages])

        context = format_docs(retriever.invoke(user_question))

        final_prompt = prompt.invoke({
            "context": context,
            "history": history_text,
            "question": user_question
        })

        response = llm.invoke(final_prompt).content

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
