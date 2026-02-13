import streamlit as st
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Policy Assistant", layout="wide")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hello 👋 Ask me anything about company policies."}
    ]

# ---------------- CSS ----------------
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"]{
    background:#0f172a;
    color:white;
    font-family: Inter, sans-serif;
}

/* HEADER */
.header{
    text-align:center;
    padding:28px 10px 22px 10px;
    border-radius:18px;
    margin-bottom:20px;
    background:linear-gradient(135deg,#020617,#020617,#071d2b);
    box-shadow:0px 0px 45px rgba(0,255,200,0.15);
}

.title{
    font-size:46px;
    font-weight:700;
    background:linear-gradient(90deg,#00ffd5,#4ade80);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    text-shadow:0 0 18px rgba(0,255,200,.35);
}

.subtitle{
    color:#94a3b8;
    margin-top:6px;
    font-size:16px;
}

/* CHAT CONTAINER */
.chat-container{
    display:flex;
    flex-direction:column;
    gap:14px;
    padding:10px 4% 120px 4%;
}

/* ROWS */
.chat-row{
    display:flex;
    width:100%;
}

.user-row{ justify-content:flex-end; }
.bot-row{ justify-content:flex-start; }

/* BUBBLES */
.user-msg{
    background:#1e293b;
    color:white;
    padding:12px 16px;
    border-radius:18px 18px 4px 18px;
    max-width:60%;
    font-size:15px;
    line-height:1.45;
    white-space:pre-wrap;
    box-shadow:0 4px 14px rgba(0,0,0,.35);
}

.bot-msg{
    background:#ffffff;
    color:#0f172a;
    padding:12px 16px;
    border-radius:18px 18px 18px 4px;
    max-width:60%;
    font-size:15px;
    line-height:1.45;
    white-space:pre-wrap;
    box-shadow:0 4px 14px rgba(0,0,0,.25);
}

/* INPUT */
.stTextInput>div>div>input{
    background:#020617;
    color:white;
    border:1px solid #334155;
    border-radius:14px;
    padding:14px;
}
.stTextInput>div>div>input::placeholder{color:#cbd5e1;}
.stTextInput>div>div>input:focus{
    border:1px solid #00ffd5;
    box-shadow:0 0 8px rgba(0,255,200,.4);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <div class="title">AI Policy Assistant</div>
    <div class="subtitle">Ask anything about company rules & benefits</div>
</div>
""", unsafe_allow_html=True)

# ---------------- DISPLAY CHAT ----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-row user-row">
            <div class="user-msg">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row bot-row">
            <div class="bot-msg">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.text_input("Message", placeholder="Message Policy Assistant...", key="input")

# ---------------- POLICY ANSWER LOGIC ----------------
def answer_question(q):

    q = q.lower()

    greetings = ["hi","hello","hey","good morning","good evening"]
    if q in greetings:
        return "Hello 👋 How can I assist you regarding company policies?"

    if "maternity" in q:
        return """Employees are encouraged to take up to 16 weeks of maternity leave.
They must inform their supervisor in writing as early as possible.
Extended leave must also be discussed with the Executive Director."""

    if "vacation" in q or "leave balance" in q:
        return "Employees should take a minimum of two weeks (10 business days) of paid vacation per year."

    if "approval" in q:
        return "You must inform your supervisor in writing in advance. Extended leave requires Executive Director discussion."

    return "Not mentioned in company policy."

# ---------------- SEND MESSAGE ----------------
if user_input:

    st.session_state.messages.append({"role":"user","content":user_input})

    # typing indicator
    with st.spinner("AI is thinking..."):
        time.sleep(0.6)
        reply = answer_question(user_input)

    st.session_state.messages.append({"role":"assistant","content":reply})

    st.rerun()
