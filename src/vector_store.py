import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "data/index"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorStore:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = get_embeddings()
        self.db: FAISS | None = None
        self._load_if_exists()

    def _load_if_exists(self):
        index_path = self.index_dir / "index.faiss"
        if index_path.exists():
            self.db = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def add_documents(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        if self.db is None:
            self.db = FAISS.from_documents(docs, self.embeddings)
        else:
            self.db.add_documents(docs)
        self.db.save_local(str(self.index_dir))
        return len(docs)

    def similarity_search(self, query: str, k: int = 6,
                          filter_source: str | None = None) -> list[Document]:
        if self.db is None:
            return []
        results = self.db.similarity_search(query, k=k * 3 if filter_source else k)
        if filter_source:
            results = [r for r in results if r.metadata.get("source_file") == filter_source]
            results = results[:k]
        return results

    def as_retriever(self, k: int = 6):
        if self.db is None:
            return None
        return self.db.as_retriever(search_kwargs={"k": k})

    def get_all_sources(self) -> list[str]:
        if self.db is None:
            return []
        docs = self.db.similarity_search("medical research", k=1000)
        seen = {}
        for doc in docs:
            sf = doc.metadata.get("source_file", "unknown")
            if sf not in seen:
                seen[sf] = doc.metadata
        return [{"source_file": k, **v} for k, v in seen.items()]

    @property
    def is_empty(self) -> bool:
        return self.db is None