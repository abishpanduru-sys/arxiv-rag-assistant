import ollama


def generate_answer(context, question):

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    context = "RAG combines information retrieval with language generation."

    answer = generate_answer(
        context,
        "What is RAG?"
    )

    print(answer)