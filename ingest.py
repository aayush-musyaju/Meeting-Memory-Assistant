"""Ingest PDF meeting notes into the vector store."""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import (
    MEETING_NOTES_DIR,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_embeddings,
)


def load_pdfs(directory: str):
    """Load all PDF files from a directory."""
    documents = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return documents
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return documents
    
    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        print(f"Loading: {filename}")
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            # Add source filename to metadata
            for doc in docs:
                doc.metadata["source_file"] = filename
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return documents


def split_documents(documents):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    """Create and persist the vector store."""
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    embeddings = get_embeddings()
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="meeting_notes",
    )
    
    return vector_store


def main():
    """Main ingestion pipeline."""
    print("=" * 50)
    print("Meeting Notes Ingestion")
    print("=" * 50)
    
    # Load PDFs
    print(f"\nLoading PDFs from: {MEETING_NOTES_DIR}")
    documents = load_pdfs(MEETING_NOTES_DIR)
    
    if not documents:
        print("\nNo documents to process. Add PDF files to data/meeting_notes/")
        return
    
    print(f"\nLoaded {len(documents)} pages from PDF files")
    
    # Split into chunks
    print("\nSplitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store
    print("\nCreating vector store...")
    create_vector_store(chunks)
    print(f"Vector store saved to: {CHROMA_DB_DIR}")
    
    print("\n" + "=" * 50)
    print("Ingestion complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
