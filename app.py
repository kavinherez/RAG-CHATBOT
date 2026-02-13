# ================= AI POLICY ASSISTANT — PRODUCTION FINAL =================

import streamlit as st
import time
import re

st.set_page_config(page_title="AI Policy Assistant", layout="wide")

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
.stChatInputContainer{padding-bottom:25px;}
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

# ================= GUARDRAIL =================
def is_policy_question(q: str) -> bool:
    allowed = [
        "leave","vacation","holiday","benefit","salary","payroll",
        "insurance","policy","work","office","attendance","remote",
        "maternity","paternity","sick","pto","hr","dress","conduct"
    ]
    q=q.lower()
    return any(word in q for word in allowed)

def is_greeting(q: str) -> bool:
    q=q.lower().strip()
    return q in ["hi","hello","hey","good morning","good afternoon","good evening"]

# ================= POLICY ENGINE =================
def get_policy_answer(q):

    # block non HR questions
    if not is_policy_question(q) and not is_greeting(q):
        return "I can only answer questions related to company policies, benefits, and workplace rules."

    # greeting
    if is_greeting(q):
        return "Hello 👋 How can I assist you regarding company policies?"

    q=q.lower()

    # known policies
    if "maternity" in q:
        return "Employees are encouraged to take up to 16 weeks of maternity leave and must inform their supervisor in writing as early as possible."

    if "vacation" in q or "pto" in q:
        return "Employees should take at least two weeks (10 business days) of paid vacation annually."

    # HR but not in KB
    return "Not mentioned in company policy."

# ================= STREAMING =================
def stream_text(text):
    words=text.split()
    out=""
    for w in words:
        out+=w+" "
        yield out
        time.sleep(0.02)

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    role=msg["role"]
    content=msg["content"]

    if role=="user":
        st.markdown(f'''
        <div class="chat-row user-row">
            <div class="user-msg">{content}</div>
        </div>
        ''', unsafe_allow_html=True)

    else:
        st.markdown(f'''
        <div class="chat-row bot-row">
            <div class="bot-msg">{content}</div>
        </div>
        ''', unsafe_allow_html=True)

# ================= INPUT =================
prompt = st.chat_input("Message Policy Assistant...")

if prompt:

    # store user msg
    st.session_state.messages.append({"role":"user","content":prompt})

    # show thinking
    thinking = st.empty()
    thinking.markdown('''
    <div class="chat-row bot-row">
        <div class="bot-msg">AI is thinking...</div>
    </div>
    ''', unsafe_allow_html=True)

    answer = get_policy_answer(prompt)
    thinking.empty()

    # streaming reply
    response_box = st.empty()
    full=""
    for chunk in stream_text(answer):
        full=chunk
        response_box.markdown(f'''
        <div class="chat-row bot-row">
            <div class="bot-msg">{full}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":full})
    st.rerun()
