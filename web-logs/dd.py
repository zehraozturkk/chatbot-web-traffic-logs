from pinecone import Pinecone, ServerlessSpec
import openai
import warnings 
from dotenv import load_dotenv
import os 
import json
import numpy as np
from uuid import uuid4
import re

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('web-log-datas')


# Ay kısaltmalarını tam adlara eşleştiren sözlük
MONTH_MAP = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December"
}

def convert_short_month_to_full(date_string):
    # Regex ile kısa ay adını bul
    pattern = r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"
    match = re.search(pattern, date_string)
    if match:
        short_month = match.group(1)  # Kısa ay adını al
        full_month = MONTH_MAP[short_month]  # Tam adını bul
        # Kısa ay adını tam adıyla değiştir
        date_string = re.sub(pattern, full_month, date_string)
    return date_string

def retrieve(query, top_k, namespace, emb_model):
    # Query'yi embed ediyoruz
    query_response = openai.Embedding.create(
        input=query,
        model=emb_model
    )
    query_emb = query_response['data'][0]['embedding']

    # Pinecone'dan sorgu yapıyoruz
    query_result = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    retrieved_docs = []
    sources = []

    # Sonuçları işliyoruz
    for result in query_result["matches"]:
        date_time_full = convert_short_month_to_full(result["metadata"]["date_time"])
        retrieved_doc = f"{result['metadata']['summary']} (Date: {date_time_full})"
        retrieved_docs.append(retrieved_doc)
        sources.append((date_time_full, result["metadata"]["url"]))

    return retrieved_docs, sources


# Test için sorgu
query = "Get methods takes which status code"

documents, sources = retrieve(
    query=query,
    top_k=20,
    namespace="changeembed_namespace",
    emb_model="text-embedding-3-small"
)

print("Query:", query)
print("Documents:", documents)
print("Sources:", sources)
