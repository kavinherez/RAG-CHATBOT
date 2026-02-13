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

def normalize_question(q: str):
    q = q.lower()
    replacements = {
        "gone": "leave",
        "away": "leave",
        "absent": "leave",
        "not coming": "leave",
        "off work": "leave",
        "time off": "leave",
        "break": "leave",
        "months": "long leave",
        "weeks": "leave",
        "personal reasons": "leave",
        "travel": "vacation",
        "holiday": "vacation",
    }
    for k, v in replacements.items():
        if k in q:
            q += " " + v
    return q

# ================= RAG ANSWER =================
def generate_ai_answer(question):

    if is_greeting(question):
        return None, "Hello 👋 I can help you understand company HR policies like leave, benefits and approvals."

    normalized_q = normalize_question(question)
    q_embedding = model.encode([normalized_q])

    scores = cosine_similarity(q_embedding, policy_embeddings)[0]
    top_indices = np.argsort(scores)[-2:][::-1]
    top_scores = scores[top_indices]

    if top_scores[0] < 0.28:
       return None, "Not mentioned in company policy."


    context_blocks = []
    for idx in top_indices:
        if scores[idx] > 0.30:
            context_blocks.append(POLICIES[idx]["text"])

    context = "\n\n".join(context_blocks)

    system_prompt = """


You are an AI HR Policy Assistant.

Answer using ONLY the provided company policy context.
Explain the policy in relation to the employee's situation.
Do NOT invent rules or recommendations beyond the policy.
Keep the answer short and clear.
"""





    user_prompt = f"""
Company Policy Context:
{context}

Employee Question:
{question}
"""

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        stream=True
    )

    return stream, None

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    role_class = "user-row" if msg["role"]=="user" else "bot-row"
    bubble_class = "user-msg" if msg["role"]=="user" else "bot-msg"
    st.markdown(f'''
    <div class="chat-row {role_class}">
        <div class="{bubble_class}">{msg["content"]}</div>
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
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_answer += delta.content
                response_box.markdown(f'''
                <div class="chat-row bot-row">
                    <div class="bot-msg">{full_answer}</div>
                </div>
                ''', unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":full_answer})
    st.session_state.thinking = False
    st.rerun()


