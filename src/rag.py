from src.retrieval.retriever import retrieve_chunks
from src.generation.llm import generate_answer


def answer_question(question):

    results = retrieve_chunks(question, top_k=5)

    context = "\n\n".join(
        result["chunk"]["text"]
        for result in results
    )

    answer = generate_answer(
        context=context,
        question=question
    )

    sources = []

    for result in results:
        chunk = result["chunk"]

        sources.append({
            "paper": chunk["paper"],
            "page": chunk["page_number"],
            "score": result["score"]
        })

    return answer, sources


if __name__ == "__main__":

    question = "What is retrieval augmented generation?"

    answer, sources = answer_question(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for source in sources:
        print(
            f"- {source['paper']} | "
            f"Page {source['page']} | "
            f"Score {source['score']:.3f}"
        )