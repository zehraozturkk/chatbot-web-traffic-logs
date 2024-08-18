from http.client import responses

from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain.embeddings import SentenceTransformerEmbeddings

from langchain_pinecone import PineconeVectorStore

import streamlit as st
from openai import embeddings

st.title("Chatbox")
st.header("hey, lets chat")
load_dotenv()

file = "nginx_logs.txt"

with open(file, "r") as file:
    content = file.read()

documents = [Document(page_content=content)]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splited_documents = text_splitter.split_documents(documents)

print(splited_documents)
# Belgeleri Chroma'ya ekleme
vectorstore = Chroma.from_documents(
    documents=splited_documents,
    embedding=OpenAIEmbeddings()
)

#pinecone

index_name = "log-index"

pinecone = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name
)

retriever = vectorstore.as_retriever()

chat = ChatOpenAI(model="gpt-3.5-turbo",  temperature = 0.4)

prompt = hub.pull("rlm/rag-prompt")




if "flowmessages" not in st.session_state:
    st.session_state['flowmessages']=[
        SystemMessage(content="You are a helpful assistant that provides detailed and accurate information. Answer questions politely and concisely.")
    ]


def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))

    # Bilgi alma işlemi
    context = retriever.get_relevant_documents(question)  # Soruya en uygun belgeleri al
    context_str = "\n".join([doc.page_content for doc in context])

    # Prompt'u kullanarak bağlam oluşturma
    formatted_prompt = prompt.format(context=context_str, question=question)

    # Chat modeline bağlamı ekleyerek yanıt oluşturma
    st.session_state['flowmessages'].append(SystemMessage(content=formatted_prompt))

    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))

    return answer.content

input_text = st.text_input("Input:", key="input")
if st.button("Ask the Question"):
    if input_text:
        response = get_chatmodel_response(input_text)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.write("Please enter a question.")
