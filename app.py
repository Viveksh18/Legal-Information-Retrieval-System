import streamlit as st
from retriever import ask_rag  

st.set_page_config(
    page_title="CRAG Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ CRAG Legal Assistant")
st.markdown("Corrective Retrieval-Augmented Generation System")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask a legal question..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.spinner("Thinking..."):
        answer = ask_rag(prompt)

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})