from pinecone import Pinecone, ServerlessSpec
import openai
import warnings 
from dotenv import load_dotenv
import os 
import json
import numpy as np
from retrieval2 import search

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('web-log-datas')


def create_enhanced_context(documents, sources):
    """
    Belgeler ve kaynaklardan zenginleştirilmiş context oluştur
    """
    enhanced_docs = []
    for doc, source in zip(documents, sources):
        # HTTP method ve status bilgisini ayıkla
        method = "Unknown"
        status = "Unknown"
        if doc.startswith("["):
            parts = doc[1:].split("]")[0].split("-")
            if len(parts) == 2:
                method = parts[0].strip()
                status = parts[1].strip()

        # Tarih bilgisini parse et
        date_parts = source.split(":")
        date = date_parts[0] if date_parts else source

        # Zenginleştirilmiş context oluştur
        enhanced_doc = f"""
Date: {date}
Method: {method}
Status: {status}
Log: {doc}
"""
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs



def prompt_with_context_builder(query, docs):
    delim = '\n\n---\n\n'
    prompt_start = """Answer the question based on the context below. The context contains web server logs with dates, HTTP methods, and status codes. If the question asks about a specific time period, only consider logs from that period. If no relevant information is found in the context, say so.
    Context:
"""
    prompt_end = f'\n\nQuestion: {query}\nAnswer: Please analyze the logs and provide a clear answer with statistics when possible.'

    prompt = prompt_start + delim.join(docs) + prompt_end
    return prompt

def question_answering(prompt, sources, chat_model):
    sys_prompt = "You are a helpful assistant that always answers questions."
    
    # Use OpenAI chat completions to generate a response
    res = openai.ChatCompletion.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    answer = res.choices[0].message.content.strip()

    return answer

def analyze_logs(query, top_k=200):
    # Belgeleri ve kaynakları al
    documents, sources = search(
        query=query,
        top_k=top_k,
        namespace="changeembed_namespace",
        emb_model="text-embedding-3-small"
    )
    
    # Zenginleştirilmiş context oluştur
    enhanced_docs = create_enhanced_context(documents, sources)
    
    # Prompt oluştur
    prompt_with_context = prompt_with_context_builder(query, enhanced_docs)

    answer = question_answering(
        prompt=prompt_with_context,
        sources=sources,
        chat_model='gpt-4-turbo-preview'  # veya kullanmak istediğiniz model
    )
    
    return answer


query = "which url takes the 500 code"
result = analyze_logs(query)
print(result)
