import os
from typing import TypedDict

from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph


class SummarySections(TypedDict):
    summary: str
    takeaways: list[str]


class SummaryState(TypedDict):
    """The data passed between nodes in the graph."""

    article_title: str
    article: str
    summary: str
    takeaways: list[str]


class SummaryRequest(TypedDict):
    article_title: str
    article: str


class SummaryResponse(TypedDict):
    article_title: str
    summary: str
    takeaways: list[str]


def build_graph(model):
    structured_model = model.with_structured_output(
        SummarySections,
        method="json_mode",
    )

    def summarize_article(state: SummaryState) -> dict[str, str | list[str]]:
        result = structured_model.invoke(
            [
                (
                    "system",
                    "Return only a valid JSON object with two keys (result in arabic with Egyptian Accent): "
                    '"summary" (a concise paragraph of at most 50 words) and '
                    '"takeaways" (an array of three to five short strings). '
                    "Use simple language and only use information from the article. "
                    "Do not return Markdown or extra keys.",
                ),
                (
                    "human",
                    f"Article title: {state['article_title']}\n\n"
                    f"Article text:\n{state['article']}",
                ),
            ]
        )
        return {
            "summary": result["summary"],
            "takeaways": result["takeaways"],
        }

    graph_builder = StateGraph(SummaryState)
    graph_builder.add_node("summarize_article", summarize_article)
    graph_builder.add_edge(START, "summarize_article")
    graph_builder.add_edge("summarize_article", END)
    return graph_builder.compile()


app = FastAPI(title="LangGraph Article Summarizer")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummaryResponse)
def summarize_article(request: SummaryRequest) -> SummaryResponse:
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="The server is missing its GROQ_API_KEY configuration.",
        )
    if not 1 <= len(request["article_title"]) <= 200:
        raise HTTPException(
            status_code=422,
            detail="article_title must be between 1 and 200 characters.",
        )
    if not 20 <= len(request["article"]) <= 30000:
        raise HTTPException(
            status_code=422,
            detail="article must be between 20 and 30000 characters.",
        )

    model = ChatGroq(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    )
    graph = build_graph(model)
    result = graph.invoke(
        {
            "article_title": request["article_title"],
            "article": request["article"],
            "summary": "",
            "takeaways": [],
        }
    )
    summary = result["summary"]
    takeaways = result["takeaways"]
    if (
        not isinstance(summary, str)
        or not isinstance(takeaways, list)
        or not 3 <= len(takeaways) <= 5
        or not all(isinstance(takeaway, str) for takeaway in takeaways)
    ):
        raise HTTPException(
            status_code=502,
            detail="Groq must return a summary and three to five takeaways.",
        )

    return SummaryResponse(
        article_title=request["article_title"],
        summary=summary,
        takeaways=takeaways,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
