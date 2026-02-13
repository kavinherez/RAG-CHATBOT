# ================= AI POLICY ASSISTANT — FINAL (STRICT GROUNDED RAG) =================

import streamlit as st
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

st.set_page_config(page_title="AI Policy Assistant", layout="wide")

# ================= GROQ CLIENT =================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ================= GLOBAL STYLE =================
st.markdown("""
<style>
.stApp { background: #f3f4f6; }
header {visibility:hidden;}
footer {visibility:hidden;}

.title-box{
    text-align:center;
    padding:35px 20px;
    border-radius:18px;
    margin-bottom:25px;
    background: radial-gradient(circle at center,#0f172a,#020617);
    box-shadow: 0 0 40px rgba(16,163,127,0.25),
                0 0 90px rgba(16,163,127,0.15);
}
.title-text{
    font-size:48px;
    font-weight:800;
    background: linear-gradient(90deg,#10a37f,#4ade80,#22d3ee);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    text-shadow:0 0 12px rgba(16,163,127,0.6);
    margin-bottom:6px;
}
.subtitle{ color:#9ca3af; font-size:18px; }

.chat-row{display:flex;width:100%;}
.user-row{justify-content:flex-end;}
.bot-row{justify-content:flex-start;}

.user-msg{
    background:#1f2937;color:white;padding:12px 16px;border-radius:16px;
    width:fit-content;max-width:55%;margin:8px 0;
}

.bot-msg{
    background:#ffffff;color:#111827;padding:12px 16px;border-radius:16px;
    width:fit-content;max-width:55%;margin:8px 0;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

.stChatInputContainer textarea{
    background:#ffffff !important;color:#111827 !important;
    caret-color:#111827 !important;border-radius:14px !important;
    border:1px solid #d1d5db !important;
}
.stChatInputContainer textarea::placeholder{color:#6b7280 !important;}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="title-box">
    <div class="title-text">AI Policy Assistant</div>
    <div class="subtitle">Ask anything about company rules & benefits</div>
</div>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hello 👋 Ask me anything about company policies."}
    ]

if "thinking" not in st.session_state:
    st.session_state.thinking = False

# ================= COMPANY POLICIES =================
POLICIES = [
    "Employees are encouraged to take up to 16 weeks of maternity leave and must inform their supervisor in writing as early as possible.",
    "Employees should take at least two weeks (10 business days) of paid vacation annually.",
    "Extended leave must be communicated to the reporting manager and may require approval from the Executive Director."
]

# ================= LOAD EMBEDDINGS =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_resource
def embed_policies():
    return model.encode(POLICIES)

policy_embeddings = embed_policies()

# ================= QUERY NORMALIZATION =================
def normalize(q):
    q=q.lower()
    synonyms={
        "gone":"leave","away":"leave","absent":"leave",
        "break":"leave","months":"long leave","weeks":"leave",
        "personal":"leave","travel":"vacation"
    }
    for k,v in synonyms.items():
        if k in q:
            q+=" "+v
    return q

# ================= RETRIEVAL =================
def retrieve_context(question):
    q = normalize(question)
    q_embed = model.encode([q])
    scores = cosine_similarity(q_embed, policy_embeddings)[0]

    # keep top 2 semantically closest chunks
    top_indices = np.argsort(scores)[-2:]

    context_chunks = []
    for i in top_indices:
        if scores[i] >= 0.28:   # medium semantic match
            context_chunks.append(POLICIES[i])

    if not context_chunks:
        return None

    return "\n".join(context_chunks)

# ================= LLM ANSWER =================
def ask_llm(context,question):

    system="""

You are an HR policy assistant.

Answer ONLY using the provided policy text.
If the policy does not explicitly contain the answer, reply exactly:

Not mentioned in company policy.

Do not guess.
Do not assume.
Do not create new rules.
"""





    user=f"""
Policy:
{context}

Question:
{question}
"""

    stream=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        stream=True
    )
    return stream

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    role="user-row" if msg["role"]=="user" else "bot-row"
    bubble="user-msg" if msg["role"]=="user" else "bot-msg"
    st.markdown(f'''
    <div class="chat-row {role}">
        <div class="{bubble}">{msg["content"]}</div>
    </div>
    ''',unsafe_allow_html=True)

# ================= INPUT =================
prompt=st.chat_input("Message Policy Assistant...")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.session_state.thinking=True
    st.rerun()

# ================= RESPONSE =================
if st.session_state.thinking:

    question=st.session_state.messages[-1]["content"]

    thinking=st.empty()
    thinking.markdown('<div class="chat-row bot-row"><div class="bot-msg">AI is thinking...</div></div>',unsafe_allow_html=True)
    time.sleep(0.5)

    context=retrieve_context(question)
    thinking.empty()

    response_box=st.empty()
    full=""

    if context is None:
        full="Not mentioned in company policy."
        response_box.markdown(f'<div class="chat-row bot-row"><div class="bot-msg">{full}</div></div>',unsafe_allow_html=True)

    else:
        stream=ask_llm(context,question)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full+=chunk.choices[0].delta.content
                response_box.markdown(f'<div class="chat-row bot-row"><div class="bot-msg">{full}</div></div>',unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":full})
    st.session_state.thinking=False
    st.rerun()


