"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

OPENAI_API_KEY = "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"   # found at platform.openai.com/account/api-keys

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def ingest_docs():
    """Get documents from web pages."""
    # We can now load those documents into memory with LangChain with 2 lines of code:
    print("Loading the documents from directory...")
    loader = DirectoryLoader(
        './data/infosys/', # my local directory
        glob='**/*.pdf'     # we only get pdfs
    )
    docs = loader.load()
    # And we split them into chunks. Each chunk will correspond to an embedding vector
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    print("Creating FAISS Indexes...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs_split, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
