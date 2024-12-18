from pinecone import Pinecone, ServerlessSpec
import warnings 
from dotenv import load_dotenv
import os 
warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)


index_name = "web-log-datas"

def create_index(index_name):
    if index_name not in pc.list_indexes():
        pc.create_index(
            name = index_name,
            dimension= 1536,
            spec=ServerlessSpec( 
                cloud='aws',
                region='us-east-1'
            )
        )
    print(f"Index '{index_name}' mevcut veya başarıyla oluşturuldu.")

create_index(index_name)
index = pc.Index(index_name)  

print(index.describe_index_stats())
