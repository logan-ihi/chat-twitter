from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

import pandas as pd
import tiktoken

from tqdm import tqdm

import os
import zipfile
from urllib.request import urlopen
from io import BytesIO

def embed_document(vector_db, splitter, document_id, document):
    metadata = [{'document_id': document_id}]
    split_documents = splitter.create_documents([str(document)], metadatas=metadata)

    texts = [d.page_content for d in split_documents]
    metadatas = [d.metadata for d in split_documents]

    docsearch = vector_db.add_texts(texts, metadatas=metadatas)

def zipfile_from_github():
    http_response = urlopen('https://github.com/twitter/the-algorithm/archive/refs/heads/main.zip')
    zf = BytesIO(http_response.read())
    return zipfile.ZipFile(zf, 'r')

def zipfile_from_local():
    # load ih-dev-static-export-filtered.zip
    zip_file = open('../ih-dev-static-export-filtered.zip', 'rb')
    return zipfile.ZipFile(zip_file, 'r')

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    openai_organization=os.environ['OPENAI_ORG_ID'],
)
encoder = tiktoken.get_encoding('cl100k_base')

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment='asia-southeast1-gcp-free'
)
vector_store = Pinecone(
    index=pinecone.Index('ih-test-index'),
    embedding_function=embeddings.embed_query,
    text_key='text',
    namespace='ih-dev-static-export-filtered'
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

total_tokens, corpus_summary = 0, []
file_texts, metadatas = [], []
with zipfile_from_local() as zip_ref:
    zip_file_list = zip_ref.namelist()
    
    pbar = tqdm(zip_file_list, desc=f'Total tokens: 0')
    for file_name in pbar:
        if (file_name.endswith('/') or 
            any(f in file_name for f in ['.DS_Store', '.gitignore']) or 
            any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])
        ):
            continue
        else:
            with zip_ref.open(file_name, 'r') as file:
                file_contents = str(file.read())
                file_name_trunc = str(file_name).replace('ih-dev-static-export-filtered/', '')
                
                n_tokens = len(encoder.encode(file_contents))
                total_tokens += n_tokens
                corpus_summary.append({'file_name': file_name_trunc, 'n_tokens': n_tokens})

                file_texts.append(file_contents)
                metadatas.append({'document_id': file_name_trunc})
                pbar.set_description(f'Total tokens: {total_tokens}')

split_documents = splitter.create_documents(file_texts, metadatas=metadatas)
vector_store.from_documents(
    documents=split_documents, 
    embedding=embeddings,
    index_name='ih-test-index',
    namespace='ih-dev-static-export-filtered'
)

pd.DataFrame.from_records(corpus_summary).to_csv('data/corpus_summary_ih.csv', index=False)