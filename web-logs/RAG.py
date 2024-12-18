from pinecone import Pinecone, ServerlessSpec
import openai
import warnings 
from dotenv import load_dotenv
import os 
import json
import numpy as np
from retrieval import retrieve

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('web-log-datas')

query = "Get methods takes which status code"


documents, sources = retrieve(
    query,
    top_k= 100,
    namespace= "changeembed_namespace",
    emb_model= "text-embedding-3-small"
)

def prompt_with_context_builder(query, docs):
    delim = '\n\n---\n\n'
    prompt_start = 'Answer the question based on the context below.\n\nContext:\n'
    prompt_end = f'\n\nQuestion: {query}\nAnswer:'

    prompt = prompt_start + delim.join(docs) + prompt_end
    return prompt

prompt_with_context = prompt_with_context_builder(query, documents)
print(prompt_with_context)


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
    answer += "\n\nSources:"
    for source in sources:
        answer += "\n" + source[0] + ": " + source[1]
    
    return answer

answer = question_answering(
  prompt=prompt_with_context,
  sources=sources,
  chat_model='gpt-4o-mini')

print(answer)