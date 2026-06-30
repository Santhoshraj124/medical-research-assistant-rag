import os
from typing import Iterator
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.vector_store import VectorStore


load_dotenv()

MODEL = "llama3.2"


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[{i}] Source: {meta.get('source_file', 'unknown')} | "
            f"Page: {meta.get('page', '?')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _get_citations(docs: list[Document]) -> list[dict]:
    return [
        {
            "index": i,
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", "?"),
        }
        for i, doc in enumerate(docs, 1)
    ]


class RAGEngine:
    def __init__(self, vector_store: VectorStore, model: str = MODEL):
        self.store = vector_store
        self.llm = ChatOllama(
            model=model,
            temperature=0.1,
        )

    def _invoke_chain(self, prompt: ChatPromptTemplate, inputs: dict) -> str:
        """Safe invoke with error handling."""
        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke(inputs)
            return result if result else "No response generated. Please try again."
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _iter(self, text: str) -> Iterator[str]:
        """Wrap a string as an iterator for streaming compatibility."""
        yield text

    # ── Q&A ───────────────────────────────────────────────────────────────────
    def stream_answer(
        self,
        question: str,
        source_filter: str | None = None,
    ) -> tuple[Iterator[str], list[dict]]:
        docs = self.store.similarity_search(question, k=6, filter_source=source_filter)

        if not docs:
            return self._iter("No relevant content found in the uploaded papers."), []

        context = _format_docs(docs)
        citations = _get_citations(docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a medical research assistant. "
             "Answer using ONLY the provided context excerpts. "
             "Cite sources as [1], [2], etc. matching the numbered excerpts. "
             "If the context does not contain the answer, say so clearly. "
             "Define medical jargon on first use. "
             "Always provide a detailed and helpful answer."),
            ("human",
             "Context:\n{context}\n\n"
             "Question: {question}\n\n"
             "Answer with citations [1], [2], etc.:"),
        ])

        answer = self._invoke_chain(prompt, {"context": context, "question": question})
        return self._iter(answer), citations

    # ── Summarise ─────────────────────────────────────────────────────────────
    def stream_summary(self, source_file: str) -> Iterator[str]:
        docs = self.store.similarity_search(
            "abstract introduction methods results conclusion findings",
            k=10,
            filter_source=source_file,
        )
        if not docs:
            return self._iter("No content found for this paper.")

        context = _format_docs(docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a medical research summariser. "
             "Produce clear, structured, accurate summaries."),
            ("human",
             "Summarise this paper based on the excerpts below.\n\n"
             "Excerpts:\n{context}\n\n"
             "Use exactly these sections:\n"
             "**Objective** — What problem does the paper address?\n"
             "**Methods** — Study design, population, interventions.\n"
             "**Key Findings** — Main results with numbers where available.\n"
             "**Conclusions** — Authors conclusions and implications.\n"
             "**Limitations** — Stated or apparent limitations.\n"
             "**Clinical Relevance** — Why this matters for practice or research."),
        ])

        result = self._invoke_chain(prompt, {"context": context})
        return self._iter(result)

    # ── Compare ───────────────────────────────────────────────────────────────
    def stream_comparison(
        self,
        source_files: list[str],
        topic: str = "",
    ) -> Iterator[str]:
        topic_q = topic or "methods findings conclusions"
        all_docs = []
        for sf in source_files:
            docs = self.store.similarity_search(topic_q, k=4, filter_source=sf)
            all_docs.extend(docs)

        if not all_docs:
            return self._iter("No content found for the selected papers.")

        context = _format_docs(all_docs)
        files_str = "\n".join(f"- {sf}" for sf in source_files)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a systematic reviewer skilled at comparing medical research studies. "
             "Identify agreements, contradictions, methodological differences, and gaps."),
            ("human",
             "Compare these papers:\n{files}\n\n"
             "Focus: {topic}\n\n"
             "Excerpts:\n{context}\n\n"
             "Use exactly these sections:\n"
             "**Overview** — Brief intro to each paper.\n"
             "**Methodology Comparison** — Designs, sample sizes, measures.\n"
             "**Findings Comparison** — Agreements and differences.\n"
             "**Strengths & Limitations** — Per paper.\n"
             "**Synthesis** — Overall conclusion from reading together.\n"
             "**Research Gaps** — Unanswered questions."),
        ])

        result = self._invoke_chain(prompt, {"files": files_str, "topic": topic_q, "context": context})
        return self._iter(result)

    # ── Literature Review ─────────────────────────────────────────────────────
    def stream_literature_review(
        self,
        topic: str,
        source_filter: list[str] | None = None,
    ) -> tuple[Iterator[str], list[dict]]:
        all_docs = []
        if source_filter:
            for sf in source_filter:
                docs = self.store.similarity_search(topic, k=4, filter_source=sf)
                all_docs.extend(docs)
        else:
            all_docs = self.store.similarity_search(topic, k=10)

        if not all_docs:
            return self._iter("No relevant content found for this topic."), []

        context = _format_docs(all_docs)
        citations = _get_citations(all_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an academic medical writer specialising in systematic literature reviews. "
             "Synthesise evidence clearly, acknowledge uncertainty, follow academic conventions."),
            ("human",
             "Write a literature review on: '{topic}'\n\n"
             "Based on these excerpts:\n{context}\n\n"
             "Use exactly these sections:\n"
             "**Introduction** — Background and importance.\n"
             "**Current Evidence** — What the literature says, cite as [1][2].\n"
             "**Contradictions & Debates** — Areas of disagreement.\n"
             "**Methodological Observations** — Patterns in study design.\n"
             "**Gaps & Future Directions** — What research is still needed.\n"
             "**Conclusion** — Summary of state of knowledge."),
        ])

        result = self._invoke_chain(prompt, {"topic": topic, "context": context})
        return self._iter(result), citations

    # ── Key Findings ──────────────────────────────────────────────────────────
    def stream_key_findings(self, source_file: str) -> Iterator[str]:
        docs = self.store.similarity_search(
            "results findings outcomes statistics p-value data",
            k=8,
            filter_source=source_file,
        )
        if not docs:
            return self._iter("No content found for this paper.")

        context = _format_docs(docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a precise medical data extractor. "
             "Never invent or assume data not present in the text."),
            ("human",
             "Extract the key findings from this paper:\n\n{context}\n\n"
             "Return organised bullet points covering:\n"
             "• Primary outcomes (with statistics and p-values where present)\n"
             "• Secondary outcomes\n"
             "• Adverse events or safety findings\n"
             "• Subgroup findings if mentioned\n"
             "• Authors main conclusion\n\n"
             "Be precise. Include numbers. Do not invent data."),
        ])

        result = self._invoke_chain(prompt, {"context": context})
        return self._iter(result)