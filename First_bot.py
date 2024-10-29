from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

st.title("Chat with llama3")

template = """Answer the question below
Here is the conversation history: {context}
Question: {question}
Answer:"""

model = OllamaLLM(model="llama3")
prompt_template = ChatPromptTemplate.from_template(template)
chain = prompt_template | model

# Initialize context in Streamlit session state
if 'context' not in st.session_state:
    st.session_state.context = ""

# Input field for the user question
user_input = st.text_input("Enter your question here and press Enter:")

if user_input and user_input.lower() != "exit1":
    with st.spinner("Generating response..."):
        result = chain.invoke({"context": st.session_state.context, "question": user_input})
        st.session_state.context += f"\nUser: {user_input}\nAI: {result}"
        st.write(f"**You**: {user_input}")
        st.write(f"**Bot**: {result}")
    st.session_state.user_input = ""  # Reset user input

# Display the conversation history
st.write("### Conversation History:")
for entry in st.session_state.context.split("\n"):
    if entry.strip():
        st.write(entry)

