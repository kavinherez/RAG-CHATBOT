# ================= AI POLICY ASSISTANT — FINAL =================

import streamlit as st
import time

st.set_page_config(page_title="AI Policy Assistant", layout="wide")

# ================= GLOBAL STYLE =================
st.markdown("""
<style>

/* Clean background */
.stApp {
    background: #f3f4f6;
}

/* Hide default header/footer */
header {visibility:hidden;}
footer {visibility:hidden;}

/* ================= TITLE ================= */

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

.subtitle{
    color:#9ca3af;
    font-size:18px;
}

/* ================= CHAT LAYOUT ================= */

.chat-row{
    display:flex;
    width:100%;
}

.user-row{
    justify-content:flex-end;
}

.bot-row{
    justify-content:flex-start;
}

.user-msg{
    background:#1f2937;
    color:white;
    padding:12px 16px;
    border-radius:16px;
    width:fit-content;
    max-width:55%;
    margin:8px 0;
}

.bot-msg{
    background:#ffffff;
    color:#111827;
    padding:12px 16px;
    border-radius:16px;
    width:fit-content;
    max-width:55%;
    margin:8px 0;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

/* ================= INPUT BOX ================= */

.stChatInputContainer{
    padding-bottom:25px;
}

.stChatInputContainer textarea{
    background:#ffffff !important;
    color:#111827 !important;
    caret-color:#111827 !important;
    border-radius:14px !important;
    border:1px solid #d1d5db !important;
}

.stChatInputContainer textarea::placeholder{
    color:#6b7280 !important;
}

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


# ================= FAKE POLICY ENGINE =================
def get_policy_answer(q):
    q = q.lower()

    if any(x in q for x in ["hi","hello","hey"]):
        return "Hello 👋 How can I assist you regarding company policies?"

    if "maternity" in q:
        return "Employees are encouraged to take up to 16 weeks of maternity leave and must inform their supervisor in writing as early as possible."

    if "vacation" in q:
        return "Employees should take at least two weeks (10 business days) of paid vacation annually."

    return "Not mentioned in company policy."


# ================= STREAMING =================
def stream_text(text):
    words = text.split()
    partial = ""
    for w in words:
        partial += w + " "
        yield partial
        time.sleep(0.03)


# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    if msg["role"] == "user":
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

    # add user message
    st.session_state.messages.append({"role":"user","content":prompt})
    st.markdown(f'''
    <div class="chat-row user-row">
        <div class="user-msg">{prompt}</div>
    </div>
    ''', unsafe_allow_html=True)

    # thinking indicator
    thinking = st.empty()
    thinking.markdown('''
    <div class="chat-row bot-row">
        <div class="bot-msg">AI is thinking...</div>
    </div>
    ''', unsafe_allow_html=True)

    time.sleep(0.7)
    answer = get_policy_answer(prompt)

    thinking.empty()

    # streaming response
    response_box = st.empty()
    full = ""
    for chunk in stream_text(answer):
        full = chunk
        response_box.markdown(f'''
        <div class="chat-row bot-row">
            <div class="bot-msg">{full}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":full})
