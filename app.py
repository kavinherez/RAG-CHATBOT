# ================= AI POLICY ASSISTANT — FINAL PRODUCTION RAG =================

import streamlit as st
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

st.set_page_config(page_title="AI Policy Assistant", layout="wide")

# ================= GROQ CLIENT =================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ================= UI STYLE =================
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
}
.title-text{
    font-size:48px;
    font-weight:800;
    background: linear-gradient(90deg,#10a37f,#4ade80,#22d3ee);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.subtitle{ color:#9ca3af; font-size:18px; }

.chat-row{display:flex;width:100%;}
.user-row{justify-content:flex-end;}
.bot-row{justify-content:flex-start;}

.user-msg{background:#1f2937;color:white;padding:12px 16px;border-radius:16px;max-width:55%;margin:8px 0;}
.bot-msg{background:#ffffff;color:#111827;padding:12px 16px;border-radius:16px;max-width:55%;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
<div class="title-text">AI Policy Assistant</div>
<div class="subtitle">Ask anything about company rules & benefits</div>
</div>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages=[{"role":"assistant","content":"Hello 👋 Ask me anything about company policies."}]
if "thinking" not in st.session_state:
    st.session_state.thinking=False

# ================= KNOWLEDGE =================
POLICIES = [
{
"search": "maternity leave pregnancy childbirth 16 weeks inform supervisor",
"display": "Employees may take up to 16 weeks of maternity leave and must inform their supervisor in writing."
},

{
"search": "vacation leave holiday annual time off two weeks 10 days",
"display": "Employees should take at least two weeks (10 business days) of vacation annually."
},

{
"search": "extended leave long leave months away personal leave long absence approval manager executive director",
"display": "Extended leave must be communicated to the reporting manager and may require Executive Director approval."
}
]

# ================= EMBEDDINGS =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_resource
def embed():
    return model.encode([p["search"] for p in POLICIES])   # FIXED

policy_embeddings = embed()

# ================= NORMALIZE QUERY =================
def normalize(q):
    q=q.lower()
    synonyms={
        "gone":"leave","away":"leave","absent":"leave",
        "months":"long leave","weeks":"leave",
        "travel":"vacation","break":"leave","personal":"leave"
    }
    for k,v in synonyms.items():
        if k in q: q+=" "+v
    return q

# ================= RETRIEVE =================
def retrieve(question):
    q=normalize(question)
    q_embed=model.encode([q])
    scores=cosine_similarity(q_embed,policy_embeddings)[0]

    best=np.array(scores).max()

    if best<0.25:
        return None

    context=[]
    for i,score in enumerate(scores):
        if score>0.30:
            context.append(POLICIES[i]["display"])   # FIXED

    return "\n".join(context) if context else None

# ================= LLM =================
def ask_llm(context,question):

    system="""
You are a company HR assistant.

Rewrite the provided policy so it answers the user's question.
Do not add advice.
Do not invent rules.
Do not explain beyond the policy.
If policy doesn't contain answer, reply exactly:
Not mentioned in company policy.
"""

    user=f"Policy:\n{context}\n\nQuestion:\n{question}"

    return client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        stream=True,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )

# ================= DISPLAY =================
for msg in st.session_state.messages:
    role="user-row" if msg["role"]=="user" else "bot-row"
    bubble="user-msg" if msg["role"]=="user" else "bot-msg"
    st.markdown(f'<div class="chat-row {role}"><div class="{bubble}">{msg["content"]}</div></div>',unsafe_allow_html=True)

prompt=st.chat_input("Message Policy Assistant...")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.session_state.thinking=True
    st.rerun()

# ================= RESPONSE =================
if st.session_state.thinking:

    q=st.session_state.messages[-1]["content"]

    thinking=st.empty()
    thinking.markdown('<div class="chat-row bot-row"><div class="bot-msg">AI is thinking...</div></div>',unsafe_allow_html=True)
    time.sleep(0.4)

    context=retrieve(q)
    thinking.empty()

    response_box=st.empty()
    full=""

    if context is None:
        full="Not mentioned in company policy."
        response_box.markdown(f'<div class="chat-row bot-row"><div class="bot-msg">{full}</div></div>',unsafe_allow_html=True)
    else:
        stream=ask_llm(context,q)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full+=chunk.choices[0].delta.content
                response_box.markdown(f'<div class="chat-row bot-row"><div class="bot-msg">{full}</div></div>',unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":full})
    st.session_state.thinking=False
    st.rerun()
