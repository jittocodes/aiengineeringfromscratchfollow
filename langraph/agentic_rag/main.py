from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader


urls = []

docs = [WebBaseLoader.load(url) for url in urls]
docs_list = [item for sublist in docs for item in sublist]