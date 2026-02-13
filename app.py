import streamlit as st
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Policy Assistant", page_icon="🤖", layout="wide")

# ---------------- BASIC STYLES ----------------
st.markdown("""
<style>
body {background-color:#0f172a;}
.user-bubble{
    background:#1f2937;
    color:white;
    padding:12px 18px;
    border-radius:16px;
    margin:8px 0;
    width:fit-content;
    margin-left:auto;
}
.assistant-bubble{
    background:#ffffff;
    color:#111827;
    padding:12px 18px;
    border-radius:16px;
    margin:8px 0;
    width:fit-content;
}
.assistant-bubble em{opacity:0.6;}
</style>
""", unsafe_allow_html=True)
# ================= HEADER =================
st.markdown("""
<div style="text-align:center; margin-top:10px; margin-bottom:30px">
    <h1 style="
        font-size:42px;
        font-weight:700;
        background: linear-gradient(90deg,#10a37f,#4ade80);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        margin-bottom:8px;
    ">
        AI Policy Assistant
    </h1>

    <p style="
        font-size:18px;
        color:#9ca3af;
        margin-top:0px;
    ">
        Ask anything about company rules & benefits
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- STREAMING RESPONSE ----------------
def stream_text(text: str):
    words = text.split(" ")
    partial = ""
    for word in words:
        partial += word + " "
        yield partial
        time.sleep(0.02)

# ---------------- TYPING INDICATOR ----------------
def show_typing_indicator(container):
    dots = ["", ".", "..", "..."]
    for i in range(6):
        container.markdown(
            f"<div class='assistant-bubble'><em>AI is thinking{dots[i % 4]}</em></div>",
            unsafe_allow_html=True
        )
        time.sleep(0.25)

# ---------------- FAKE RAG ANSWER (REPLACE WITH YOUR MODEL) ----------------
def get_answer(query: str):
    query_lower = query.lower()

    if "maternity" in query_lower:
        return "Employees are encouraged to take up to 16 weeks of maternity leave. Inform your supervisor in writing as early as possible."
    elif "vacation" in query_lower:
        return "Employees should take at least 10 business days of paid vacation annually according to policy."
    elif "hello" in query_lower or "hi" in query_lower:
        return "Hello 👋 Ask me anything about company policies."
    else:
        return "Not mentioned in company policy."

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;color:white;'>Policy Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Ask anything about company rules & benefits</p>", unsafe_allow_html=True)

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Message Policy Assistant...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)

    # Thinking indicator
    thinking_placeholder = st.empty()
    show_typing_indicator(thinking_placeholder)

    # Generate response
    response = get_answer(user_input)
    thinking_placeholder.empty()

    # Stream response
    stream_placeholder = st.empty()
    final_response = ""

    for chunk in stream_text(response):
        stream_placeholder.markdown(
            f"<div class='assistant-bubble'>{chunk}</div>",
            unsafe_allow_html=True
        )
        final_response = chunk

    # Save final response
    st.session_state.messages.append({"role": "assistant", "content": final_response})

