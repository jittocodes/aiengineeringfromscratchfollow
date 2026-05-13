import asyncio
import os
import ssl
from typing import Dict, Any, List

import certifi
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

# Configure SSL context to use the certif certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Here the chunk size is to handle rate limit
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10
)

vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],
    embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def main():
    """Main async function"""
    
    res = tavily_crawl.invoke({
        "url" : "https://python.langchain.com/",
        "max_depth" : 1,
        "extract_depth" : "advanced"
    })
    
    res_docs = res["results"]
    
    all_docs = [Document(page_content=result['raw_content'], metadata={"source" : result['url']}) for result in res_docs]
    
    print("Created the documents")
    
    print("Splitting the documents")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4000,
        chunk_overlap = 200
    )
    split_docs = splitter.split_documents(all_docs)
    
    log_success("Chunking success")
    log_header("Starting embeddings")
    
    await index_documents_async(documents=split_docs, batch_size = 500)
    
    log_success("Embeddings added in vector store")
    
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches async"""
    
    log_header("Vector Storage Phase")
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    
    async def add_batch(batch: List[Document], batch_num: int):
        try: 
            await vectorstore.aadd_documents(batch)
            log_success(f"Batch {batch_num} added with size {len(batch)}")
        except Exception as e:
            log_error(f"Batch for {batch_num} failed")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_batches = sum(1 for result in results if result is True)
    
    if successful_batches == len(batches):
        log_success("All batches added successfully")
    else:
        log_warning(f"{successful_batches} out of {len(batches)} batches added successfully")
    
if __name__ == "__main__":
    asyncio.run(main())