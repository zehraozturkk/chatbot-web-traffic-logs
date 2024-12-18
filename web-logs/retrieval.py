from pinecone import Pinecone, ServerlessSpec
import openai
import warnings
from dotenv import load_dotenv
import os
import re
import numpy as np
from uuid import uuid4
from datetime import datetime

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('web-log-datas')

# Map for converting short month names to full month names
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

# Function to convert specific dates in metadata to full month names
def convert_specific_date_to_full(date_string):
    pattern = r"(\d{1,2})(st|nd|rd|th)?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    match = re.search(pattern, date_string)
    if match:
        day = match.group(1)
        short_month = match.group(3)
        full_month = MONTH_MAP[short_month]
        date_string = re.sub(pattern, f"{day} {full_month}", date_string)
    return date_string

# Function to normalize dates in the query
def normalize_date_in_query(query):
    # Updated regex to handle full month names and abbreviations
    pattern = r"(\d{1,2})\s*(January|Feb|February|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec)\s*(\d{4})?"
    
    def replace_with_normalized(match):
        day = match.group(1)
        month = match.group(2)
        year = match.group(3) or str(datetime.now().year)
        
        # Convert full month names to their abbreviations
        month_abbr = month[:3].capitalize() if month not in MONTH_MAP else MONTH_MAP[month]
        
        return f"{day}/{month_abbr}/{year}"

    return re.sub(pattern, replace_with_normalized, query, flags=re.IGNORECASE)

# Function to retrieve documents from Pinecone
def retrieve(query, top_k, namespace, emb_model):
    # Normalize query date
    query = normalize_date_in_query(query)
    print(f"Normalized Query: {query}")

    query_date = query.split('/') if '/' in query else None


    # Embed the query
    query_response = openai.Embedding.create(
        input=query,
        model=emb_model
    )
    query_emb = query_response['data'][0]['embedding']

    # Query Pinecone index
    query_result = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    # Process results
    retrieved_docs = []
    sources = []
    for result in query_result["matches"]:
        # Dokümandaki tarihi parse etme
        doc_date = result["metadata"]["date_time"]
        
        # Tarih kontrolü - aynı ay ve yıl
        if query_date:
            doc_parts = doc_date.split('/')
            if (query_date[1] == doc_parts[1]) and (query_date[2].split(':')[0] == doc_parts[2].split(':')[0]):
                date_time_full = convert_specific_date_to_full(doc_date)
                retrieved_doc = f"{result['metadata']['summary']} (Date: {date_time_full})"
                retrieved_docs.append(retrieved_doc)
                sources.append((date_time_full, result["metadata"]["url"]))

    return retrieved_docs, sources
# Test the retrieve function
query = "500 code in september"

documents, sources = retrieve(
    query=query,
    top_k=20,
    namespace="changeembed_namespace",
    emb_model="text-embedding-3-small"
)

print("Documents:", documents)
print("Sources:", sources)
