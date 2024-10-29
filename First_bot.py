from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

st.title("Say hey to the chatbot")

template = """Answer the question below
Here is the conversation history: {context}
Question: {question}
Answer:"""

# Initialize the Ollama model without passing the port argument incorrectly
model = OllamaLLM(model="llama3")
prompt_template = ChatPromptTemplate.from_template(template)
chain = prompt_template | model

# Text area for user input
user_input = st.text_area("Enter your question here:", height=100)

if user_input:
    with st.spinner("Generating response..."):
        if 'context' not in st.session_state:
            st.session_state.context = ""
        result = chain.invoke({"context": st.session_state.context, "question": user_input})
        st.session_state.context += f"\nUser: {user_input}\nAI: {result}"
        st.write(f"**You**: {user_input}")
        st.write(f"**Bot**: {result}")

# Display the conversation history
st.write("### Conversation History:")
for entry in st.session_state.context.split("\n"):
    if entry.strip():
        st.write(entry)
