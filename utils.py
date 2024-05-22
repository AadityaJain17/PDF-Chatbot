# Function to push embedded data to Vector Store - Pinecone
import streamlit as st 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import Pinecone as PineconeStore
from dotenv import load_dotenv
from pinecone import Pinecone
import asyncio

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def push_to_pinecone(pinecone_environment,pinecone_index_name,embeddings,docs):
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs)
    pinecone = Pinecone(
        environment=pinecone_environment
        )
    # create a vectorstore from the chunks
    vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name=pinecone_index_name)

# push the vectorstore to the pinecone index
def get_vectorstore():
    vector_store = PineconeStore.from_existing_index(index_name="hr",embedding=embeddings)
    return vector_store

# get context from vector store
def get_context_retriever_chain(vector_store):
    llm = OpenAI()
    retriever = vector_store.as_retriever() 
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # If there is no chat_history, then the input is just passed directly to the retriever. 
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

#RAG chain is defined  
def get_conversational_rag_chain(retriever_chain): 
    llm = OpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "you are the chatmodel of the given document, your name is 'HR bot', you can answerr in the very familier and good manner,Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    # for passing a list of Documents to a model.
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# get responses from vector store
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']