from pinecone import Pinecone, ServerlessSpec
import openai
import warnings 
from dotenv import load_dotenv
import os 
import json
import numpy as np
from uuid import uuid4

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('web-log-datas')



with open("nginx_logs.json","r") as file:
    jsonData = json.load(file)


batch_limit = 100

for batch in np.array_split(jsonData, len(jsonData) / batch_limit):

    # Extract the metadata from each row
    metadatas = [
        {
            "date_time": entry['date_time'],
            "method": entry['method'],
            "url": entry['url'],
            "status": entry['status'],
            "referer": entry['referer'],
            "device": entry['device'],
            "summary": entry['summary']
        } 
        for entry in batch
    ]
    texts = [
        f"Datetime: {entry["date_time"]},Method: {entry['method']}, Status: {entry['status']}, URL: {entry['url']}, Summary: {entry['summary']}"
        for entry in batch
    ]

    ids = [str(uuid4()) for _ in range(len(batch))]

    response = openai.Embedding.create(input=texts, model = "text-embedding-3-small")
    embeds = [np.array(x.embedding) for x in response.data]

    index.upsert(vectors=zip(ids, embeds, metadatas),
                 namespace="changeembed_namespace")

print("Data has been successfully upserted into Pinecone.")
