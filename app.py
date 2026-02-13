# ================= AI POLICY ASSISTANT — FINAL (RAG + GROQ STREAMING) =================

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

/* TITLE */
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

/* CHAT */
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

/* INPUT */
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
    {"title":"Maternity Leave",
     "text":"Employees are encouraged to take up to 16 weeks of maternity leave and must inform their supervisor in writing as early as possible."},

    {"title":"Paid Vacation",
     "text":"Employees should take at least two weeks (10 business days) of paid vacation annually."},

    {"title":"Approvals",
     "text":"Extended leave must be communicated to the reporting manager and may require approval from the Executive Director."},

    {"title":"Scope",
     "text":"The assistant answers only HR policies, employee benefits and workplace rules."}
]

# ================= LOAD EMBEDDING MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_resource
def embed_policies():
    return model.encode([p["text"] for p in POLICIES])

policy_embeddings = embed_policies()

# ================= HELPERS =================
def is_greeting(q):
    return q.lower().strip() in ["hi","hello","hey","good morning","good afternoon","good evening"]

# ================= GROQ RAG ANSWER =================
def generate_ai_answer(question):

    if is_greeting(question):
        return None, "Hello 👋 I can help you understand company HR policies like leave, benefits and approvals."

    q_embedding = model.encode([question])
    scores = cosine_similarity(q_embedding, policy_embeddings)[0]
    best_idx = np.argmax(scores)
    confidence = scores[best_idx]

    if confidence < 0.35:
        return None, "I can only answer company policy related questions."

    context = POLICIES[best_idx]["text"]

    system_prompt = """
You are an AI HR Policy Assistant.

STRICT RULES:
- Answer ONLY using the provided company policy
- If information missing, say: Not mentioned in company policy.
- Do NOT assume anything
- Keep answer clear and professional
"""

    user_prompt = f"""
Company Policy:
{context}

User Question:
{question}
"""

    stream = client.chat.completions.create(
       model="llama-3.1-8b-instant",
        temperature=0,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        stream=True
    )

    return stream, None

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    if msg["role"]=="user":
        st.markdown(f'''
        <div class="chat-row user-row">
            <div class="user-msg">{msg["content"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="chat-row bot-row">
            <div class="bot-msg">{msg["content"]}</div>
        </div>
        ''', unsafe_allow_html=True)

# ================= INPUT =================
prompt = st.chat_input("Message Policy Assistant...")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.session_state.thinking = True
    st.rerun()

# ================= RESPONSE HANDLER =================
if st.session_state.thinking:

    last_user = st.session_state.messages[-1]["content"]

    thinking_box = st.empty()
    thinking_box.markdown('''
    <div class="chat-row bot-row">
        <div class="bot-msg">AI is thinking...</div>
    </div>
    ''', unsafe_allow_html=True)

    time.sleep(0.6)

    stream, fallback = generate_ai_answer(last_user)
    thinking_box.empty()

    response_box = st.empty()
    full_answer = ""

    if fallback:
        full_answer = fallback
        response_box.markdown(f'''
        <div class="chat-row bot-row">
            <div class="bot-msg">{full_answer}</div>
        </div>
        ''', unsafe_allow_html=True)

    else:
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_answer += token
                response_box.markdown(f'''
                <div class="chat-row bot-row">
                    <div class="bot-msg">{full_answer}</div>
                </div>
                ''', unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":full_answer})
    st.session_state.thinking = False
    st.rerun()

