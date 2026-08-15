"""Interactive Sophia AI Search CLI."""

import asyncio
import sys
from dotenv import load_dotenv

from sophia import ModelTier, SophiaEngine

load_dotenv()


async def stream_sophia_cli(query: str, tier: ModelTier = ModelTier.BALANCED):
    engine = SophiaEngine()

    print("\n" + "=" * 70)
    print(f"🔍 Question: {query}")
    print("=" * 70)

    planned_query = ""
    sources = []
    follow_ups = []

    async for event in engine.run_pipeline(user_question=query):
        if event.event_type == "status":
            pass
        elif event.event_type == "query_rewritten":
            planned_query = event.data
            print(f"🎯 Search Query: \"{planned_query}\"")
        elif event.event_type == "sources":
            if isinstance(event.data, list):
                sources = event.data
                print(f"🌐 Found {len(sources)} sources. Synthesizing answer...\n")
                print("-" * 70)
        elif event.event_type == "token":
            print(event.data, end="", flush=True)
        elif event.event_type == "follow_ups":
            if isinstance(event.data, list):
                follow_ups = event.data

    print("\n" + "-" * 70)

    # Print citation sources
    if sources:
        print("\n📚 Sources Cited:")
        for s in sources:
            domain_tag = f" `{s.domain}`" if s.domain else ""
            print(f"  [{s.index}] {s.title}{domain_tag}\n      {s.url}")

    # Print follow-up questions
    if follow_ups:
        print("\n💡 Related Questions:")
        for q in follow_ups:
            print(f"  • {q}")
    print("=" * 70 + "\n")


async def interactive_mode():
    print("\n" + "=" * 70)
    print("  🚀 SOPHIA - AI SEARCH & DISCOVERY ENGINE")
    print("  Type your question or 'exit' / 'quit' to quit.")
    print("=" * 70)

    while True:
        try:
            query = input("\n👉 Ask anything: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("\nGoodbye! 👋")
                break
            await stream_sophia_cli(query)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        asyncio.run(stream_sophia_cli(query))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
