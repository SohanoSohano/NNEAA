import arxiv
import json
import os

def scrape_arxiv(query="neural network architecture", max_results=1000):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published.isoformat(),
        })

    os.makedirs("data/research_papers", exist_ok=True)
    with open("data/research_papers/arxiv_papers.json", "w") as f:
        json.dump(papers, f, indent=4)

if __name__ == "__main__":
    scrape_arxiv()
