from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

import streamlit as st
from sympy.physics.units import temperature

st.title("Chatbox")
st.header("hey, lets chat")
load_dotenv()


llm = ChatOpenAI(model="gpt-3.5-turbo",  temperature = 0.4)

file = "nginx_logs.txt"

with open(file, "r") as file:
    content = file.read()

documents = [Document(page_content=content)]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splited_documents = text_splitter.split_documents(documents)

# Belgeleri Chroma'ya ekleme
vectorstore = Chroma.from_documents(
    documents=splited_documents,
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()  #veri tabanından bilgileri geri almak
prompt = hub.pull("rlm/rag-prompt")


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
