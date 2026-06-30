import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent))
from src.pdf_loader import load_and_split
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

console = Console()
STORE_DIR = "data/index"


def get_store_engine():
    store = VectorStore(STORE_DIR)
    return store, RAGEngine(store)


def cmd_ingest(args):
    store, _ = get_store_engine()
    for pdf_path in args.pdf:
        path = Path(pdf_path)
        if not path.exists():
            console.print(f"[red]Not found: {pdf_path}[/red]")
            continue
        with console.status(f"Indexing {path.name}…"):
            docs = load_and_split(str(path))
            added = store.add_documents(docs)
        console.print(f"[green]✓[/green] {path.name} — {added} chunks indexed")


def cmd_list(args):
    store, _ = get_store_engine()
    sources = store.get_all_sources()
    if not sources:
        console.print("[yellow]No papers indexed yet.[/yellow]")
        return

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold blue")
    t.add_column("File", width=50)
    t.add_column("Pages", justify="right")
    for s in sources:
        t.add_row(s["source_file"], str(s.get("page", "?")))
    console.print(t)


def cmd_ask(args):
    _, engine = get_store_engine()
    question = " ".join(args.question)
    console.print(Panel(f"[bold]{question}[/bold]", title="Question", border_style="blue"))

    stream, citations = engine.stream_answer(question)
    console.print(Markdown("".join(stream)))

    if citations:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        for c in citations:
            console.print(f"  [{c['index']}] {c['source_file']} | Page {c['page']}")


def cmd_summarise(args):
    store, engine = get_store_engine()
    sources = store.get_all_sources()
    if not sources:
        console.print("[yellow]No papers indexed.[/yellow]")
        return

    if args.source:
        source_file = args.source
    else:
        console.print("Available papers:")
        for s in sources:
            console.print(f"  {s['source_file']}")
        source_file = console.input("\nEnter filename: ").strip()

    stream = engine.stream_summary(source_file)
    console.print(Markdown("".join(stream)))


def cmd_compare(args):
    _, engine = get_store_engine()
    stream = engine.stream_comparison(args.files, topic=args.topic or "")
    console.print(Markdown("".join(stream)))


def cmd_review(args):
    _, engine = get_store_engine()
    topic = " ".join(args.topic)
    stream, citations = engine.stream_literature_review(topic)
    console.print(Markdown("".join(stream)))
    if citations:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        for c in citations:
            console.print(f"  [{c['index']}] {c['source_file']} | Page {c['page']}")


def cmd_findings(args):
    store, engine = get_store_engine()
    sources = store.get_all_sources()
    if not sources:
        console.print("[yellow]No papers indexed.[/yellow]")
        return

    if args.source:
        source_file = args.source
    else:
        console.print("Available papers:")
        for s in sources:
            console.print(f"  {s['source_file']}")
        source_file = console.input("\nEnter filename: ").strip()

    stream = engine.stream_key_findings(source_file)
    console.print(Markdown("".join(stream)))


def main():
    parser = argparse.ArgumentParser(
        description="Medical RAG CLI — LangChain + Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py ingest paper1.pdf paper2.pdf
  python cli.py list
  python cli.py ask What are the primary outcomes?
  python cli.py summarise --source paper1.pdf
  python cli.py compare paper1.pdf paper2.pdf --topic efficacy and safety
  python cli.py review GLP-1 agonists in type 2 diabetes
  python cli.py findings --source paper1.pdf
        """,
    )

    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("ingest", help="Index PDF papers")
    p.add_argument("pdf", nargs="+", help="PDF file paths")

    sub.add_parser("list", help="List indexed papers")

    p = sub.add_parser("ask", help="Ask a research question")
    p.add_argument("question", nargs="+")

    p = sub.add_parser("summarise", help="Summarise a paper")
    p.add_argument("--source", help="PDF filename from 'list'")

    p = sub.add_parser("compare", help="Compare papers")
    p.add_argument("files", nargs="+", help="PDF filenames to compare")
    p.add_argument("--topic", help="Comparison focus")

    p = sub.add_parser("review", help="Generate literature review")
    p.add_argument("topic", nargs="+")

    p = sub.add_parser("findings", help="Extract key findings")
    p.add_argument("--source", help="PDF filename from 'list'")

    args = parser.parse_args()
    dispatch = {
        "ingest": cmd_ingest,
        "list": cmd_list,
        "ask": cmd_ask,
        "summarise": cmd_summarise,
        "compare": cmd_compare,
        "review": cmd_review,
        "findings": cmd_findings,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()