import os
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message

import time



load_dotenv()

st.title("My Chatbox")
st.header("Hey, let's ask what do you want")

file_path = "nginx_cleaned_logs.csv"

loader = CSVLoader(file_path=file_path)
data = loader.load()

documents = [Document(page_content=str(row)) for row in data]  # Convert each row to a string

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_documents = text_splitter.split_documents(documents)

# Initialize the LLM (Language Model)
llm = ChatOpenAI(model="gpt-4", temperature=0.1)

embeddings = OpenAIEmbeddings()

# Pinecone Vector Store
index_name = "log-index-2"
pinecone_vectorstore = PineconeVectorStore.from_documents(
    documents=splitted_documents,
    embedding=embeddings,
    index_name=index_name
)

# Use the retriever (from Pinecone in this case)
retriever = pinecone_vectorstore.as_retriever()

# Load the RAG Prompt from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

# Session State for managing chat history and memory
if "flowmessages" not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a helpful assistant that provides detailed and accurate information. Answer questions politely and concisely.")
    ]

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# Prompt Templates for LLM interaction
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know'"""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template]
)

def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))

    context = retriever.get_relevant_documents(question)
    context_str = "\n".join([doc.page_content for doc in context])

    formatted_prompt = prompt_template.format_messages(
        history=st.session_state.buffer_memory.chat_memory.messages,
        input=question
    )

    st.session_state['flowmessages'].append(SystemMessage(content=context_str))

    # Get the response from the LLM
    answer = llm(messages=st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))

    # Ensure 'requests' and 'responses' exist in session_state
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []


    st.session_state['requests'].append(question)  # Append the question
    st.session_state['responses'].append(answer.content)  # Append the response

    return answer.content

# User interface for interacting with the chatbot
input_text = st.text_input("Input:", key="input")

if st.button("Ask the Question"):
    if input_text:
        response = get_chatmodel_response(input_text)
    else:
        st.write("Please enter a question.")

# Display the conversation
if st.session_state['responses']:
    for i in range(len(st.session_state['responses'])):
        if i < len(st.session_state['requests']):
            message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['responses'][i], key=str(i))


def measure_response_time(question, model, prompt_template, retriever):
    start_time = time.time()

    # Retrieve relevant documents
    context = retriever.get_relevant_documents(question)
    context_str = "\n".join([doc.page_content for doc in context])

    # Format the prompt and get the model response
    formatted_prompt = prompt_template.format_messages(
        history=st.session_state.buffer_memory.chat_memory.messages,
        input=question
    )

    answer = model(messages=[SystemMessage(content=context_str), HumanMessage(content=question)])

    end_time = time.time()

    response_time = end_time - start_time
    return answer.content, response_time


# Example usage
response, response_time = measure_response_time(input_text, llm, prompt_template, retriever)
st.write(f"Response Time: {response_time:.2f} seconds")


