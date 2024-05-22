import streamlit as st
from utils import *

load_dotenv()

# loader = PyPDFLoader('/Users/aadityajain/Desktop/pdf_chatbot/HR Policy_CB.pdf')
# docs = loader.load()
# push_to_pinecone("gcp-starter", "hr", embeddings, docs)

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, how may i help you?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()  

# Render selected section
st.header('Ask Anything about HR Policies')
st.text('VKAPS It Solutions Pvt Ltd.')

# conversation
user_query = st.chat_input("Ask your query here About the Given PDF...")
for message in st.session_state.chat_history :
    if isinstance(message, HumanMessage)  :
        with st.chat_message("You")   :
            st.markdown(message.content)
    else  :
        with st.chat_message("AI"):
            st.markdown(message.content)

if user_query:
    response = get_response(user_query)
    # Display user's question
    with st.chat_message("You"):
        st.markdown(user_query)
    # Display AI's answer
    with st.chat_message("AI"):
        st.markdown(response)

    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))