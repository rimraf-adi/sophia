"""Search results formatters for LLM prompt injection and Perplexity citation pipelines."""

from __future__ import annotations

from sophia.search.models import SearchResponse, SearchResult


def format_search_results_for_llm(
    response_or_results: SearchResponse | list[SearchResult],
    max_results: int | None = None,
) -> str:
    """Format search results into a clean context block for LLM prompts.

    Example output:
    ```
    Source [1]: Python 3.13 Released (python.org)
    URL: https://python.org/...
    Content: Python 3.13 is the latest version...

    Source [2]: What's new in Python (realpython.com)
    URL: https://realpython.com/...
    Content: A summary of features...
    ```
    """
    if isinstance(response_or_results, SearchResponse):
        results = response_or_results.results
    else:
        results = response_or_results

    target = results[:max_results] if max_results else results
    if not target:
        return "No relevant search results found."

    blocks = []
    for item in target:
        domain_tag = f" ({item.domain})" if item.domain else ""
        header = f"Source [{item.index}]: {item.title}{domain_tag}"
        body = f"URL: {item.url}\nContent: {item.snippet}"
        if item.published_date:
            body = f"Date: {item.published_date}\n{body}"
        blocks.append(f"{header}\n{body}")

    return "\n\n".join(blocks)


def format_sources_markdown(
    response_or_results: SearchResponse | list[SearchResult],
) -> str:
    """Format search results as a markdown list of clickable sources for the user."""
    if isinstance(response_or_results, SearchResponse):
        results = response_or_results.results
    else:
        results = response_or_results

    if not results:
        return "*No sources found.*"

    lines = ["### Sources:"]
    for r in results:
        domain = f" `{r.domain}`" if r.domain else ""
        lines.append(f"- **[{r.index}]** [{r.title}]({r.url}){domain}")

    return "\n".join(lines)


PERPLEXITY_SYSTEM_PROMPT = """You are an accurate, fast, and insightful AI search assistant (like Perplexity AI).
You answer user questions using the provided web search context.

Follow these strict rules:
1. Ground your answer in the provided search results.
2. Cite your sources inline using bracketed numbers like [1], [2], or [1][3] immediately after relevant facts or claims.
3. If different sources offer contrasting views or updates, synthesize them clearly.
4. If the search results do not contain enough information to fully answer, state what is known from sources and what is uncertain.
5. Format your response cleanly with markdown headings, bullet points, and code blocks where applicable.
"""
