from dotenv import load_dotenv
load_dotenv()

"""
    Simple process:
    1. Load the documents
    2. Split the data
    3. Embed the chunks
    4. Store the embedded chunks
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

loader = TextLoader("langchain/rag/mediublog1.txt", encoding='utf-8')
document = loader.load()

print("splitting...")

text_splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

texts = text_splitter.split_documents(document)
print(f"Created chunks with {len(texts)} chunks")

embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))

print("Ingesting....")

PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])

print("finished")