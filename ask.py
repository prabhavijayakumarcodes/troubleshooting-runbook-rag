"""Command-line on-call assistant over the Contoso troubleshooting runbook.

Usage:
    python ask.py "payment-gateway is returning ERR-504, what do I do?"
    python ask.py                 # interactive mode, empty line or Ctrl-D to quit
    python ask.py --rebuild ...   # force re-embedding the PDF
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import runbook_rag as rr  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Contoso troubleshooting runbook.")
    parser.add_argument("question", nargs="*", help="question, alert text or log line")
    parser.add_argument("-k", type=int, default=rr.TOP_K, help="chunks to retrieve (default %(default)s)")
    parser.add_argument("--semantic-only", action="store_true", help="disable error-code / service matching")
    parser.add_argument("--no-sources", action="store_true", help="hide retrieved sources")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the FAISS index")
    args = parser.parse_args()

    store = rr.load_or_build_vector_store(rr.make_embeddings(), rebuild=args.rebuild)
    chain = rr.build_rag_chain(rr.RunbookRetriever(store, k=args.k, hybrid=not args.semantic_only), rr.make_llm())

    def answer(question: str) -> None:
        out = chain.invoke({"question": question})
        print("\n" + out["answer"].strip())
        if not args.no_sources:
            print("\nSources:\n" + rr.format_sources(out["hits"]))
        print()

    if args.question:
        answer(" ".join(args.question))
        return
    print("Contoso runbook assistant. Paste a question, alert or log line (empty line to quit).")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break
        answer(question)


if __name__ == "__main__":
    main()
