import arxiv
import requests
from pathlib import Path


def download_papers(query, max_results=10):
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    for result in client.results(search):

        paper_id = result.get_short_id().replace("/", "_")
        filename = output_dir / f"{paper_id}.pdf"
        if filename.exists():
            print(f"Already exists: {filename.name}")
            continue
        

        print(f"Downloading: {result.title}")

        response = requests.get(result.pdf_url)
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)


if __name__ == "__main__":
    download_papers(
    query="retrieval augmented generation",
    max_results=20
)