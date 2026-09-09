import streamlit as st

from src.rag import answer_question


st.title("📚 arXiv RAG Assistant")

question = st.text_input("Ask a question about the research papers:")

if st.button("Ask") and question:

    with st.spinner("Searching papers and generating answer..."):
        answer, sources = answer_question(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for source in sources:
        st.write(
            f"- {source['paper']} — "
            f"Page {source['page']} "
            f"(score: {source['score']:.3f})"
        )