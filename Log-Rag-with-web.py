import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from  langchain_text_splitters import  RecursiveCharacterTextSplitter
from  langchain_openai import ChatOpenAI


load_dotenv()

loader = WebBaseLoader(
    web_paths=("https://raw.githubusercontent.com/linuxacademy/content-elastic-log-samples/master/access.log",)
)

docs = loader.load()

vectorstore =Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings()) #db'ye attık
retriever = vectorstore.as_retriever()



#daha okunabilir hale
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)
