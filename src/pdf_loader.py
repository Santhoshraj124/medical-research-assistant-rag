from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_split(filepath: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """Load a PDF and split into overlapping chunks using LangChain."""
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    # Inject source filename into metadata
    filename = Path(filepath).name
    for doc in pages:
        doc.metadata["source_file"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))

    return chunks