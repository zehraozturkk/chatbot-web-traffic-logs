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

MONTH_MAP = {
    "january": "Jan", "february": "Feb", "march": "Mar",
    "april": "Apr", "may": "May", "june": "Jun",
    "july": "Jul", "august": "Aug", "september": "Sep",
    "october": "Oct", "november": "Nov", "december": "Dec"
}

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('web-log-datas')

def extract_http_filters(query):
    HTTP_METHODS = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'}
    filters = {}
    clean_query = query.upper()

    # HTTP Method filtreleme
    for method in HTTP_METHODS:
        if method in clean_query:
            filters["method"] = method
            clean_query = clean_query.replace(method, "")

    # Status Code filtreleme
    status_code_pattern = r'\b([1-5][0-9]{2})\b'
    status_codes = re.findall(status_code_pattern, clean_query)
    if status_codes:
        filters["status_code"] = int(status_codes[0])  # İlk bulunan status code'u al
        clean_query = re.sub(status_code_pattern, "", clean_query)

    clean_query = clean_query.lower().strip()
    return filters, clean_query

def extract_date_filters(query):
    """Extract date-related filters from the query"""
    date_filters = {}
    
    # Month patterns (both full and abbreviated)
    month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
    
    # Find month in query
    month_match = re.search(month_pattern, query.lower())
    if month_match:
        month = month_match.group(1)
        # Convert full month names to abbreviated version
        if month in MONTH_MAP:
            date_filters['month'] = MONTH_MAP[month]
        elif len(month) == 3:
            date_filters['month'] = month.capitalize()
            
    # Year pattern
    year_pattern = r'\b(20\d{2})\b'
    year_match = re.search(year_pattern, query)
    if year_match:
        date_filters['year'] = year_match.group(1)
        
    # Day pattern
    day_pattern = r'\b(\d{1,2})(st|nd|rd|th)?\b'
    day_match = re.search(day_pattern, query)
    if day_match:
        date_filters['day'] = day_match.group(1)
    
    return date_filters
def matches_date_filter(metadata_date, date_filters):
    """Check if metadata date matches the date filters"""
    try:
        # Parse the metadata date format "10/Oct/2024:21:21:40 +0300"
        date_str = metadata_date.split(':')[0]  # Get "10/Oct/2024" part
        day, month, year = date_str.split('/')
        
        if 'month' in date_filters and month != date_filters['month']:
            return False
        if 'year' in date_filters and year != date_filters['year']:
            return False
        if 'day' in date_filters and day != date_filters['day']:
            return False
        return True
    except:
        return False

def search(top_k, query, emb_model, namespace):
    try:
        # Extract both HTTP and date filters
        http_filters, clean_query = extract_http_filters(query)
        date_filters = extract_date_filters(clean_query)
        
        # Create embedding for the cleaned query
        query_response = openai.Embedding.create(
            input=clean_query,
            model=emb_model
        )
        query_emb = query_response['data'][0]['embedding']

        # Prepare Pinecone filters
        filter_dict = {}
        if http_filters:
            if "method" in http_filters:
                filter_dict["method"] = {"$eq": http_filters["method"]}
            if "status_code" in http_filters:
                filter_dict["status"] = {"$eq": http_filters["status_code"]}

        # Query Pinecone
        docs = index.query(
            vector=query_emb,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=filter_dict
        )

        # Process results and apply date filtering
        retrieved_docs = []
        sources = []
        for doc in docs.get('matches', []):
            metadata = doc['metadata']
            date_time = metadata.get("date_time", "")
            
            # Apply date filters
            if date_filters and not matches_date_filter(date_time, date_filters):
                continue
                
            summary = metadata.get('summary', '')
            method = metadata.get('method', 'N/A')
            status = metadata.get('status_code', 'N/A')
            summary_with_info = f"[{method} - {status}] {summary}"
            
            retrieved_docs.append(summary_with_info)
            sources.append(date_time)

        return retrieved_docs, sources
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []
    
