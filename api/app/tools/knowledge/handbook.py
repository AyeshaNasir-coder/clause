from langchain.tools import tool
from langchain_qdrant import QdrantVectorStore


def file_search(store: QdrantVectorStore, top_k: int):

    @tool(response_format="content_and_artifact")
    def search_handbook(question: str):
        """Search the Clause shop handbook for refunds, delivery,
        payments, COD, warranty, exchanges, tracking, loyalty
        and support hours."""

        results = store.similarity_search_with_score(
            query=question,
            k=top_k,
        )

        if not results:
            return "Nothing found.", []

        text = "\n\n".join(doc.page_content for doc, _ in results)  # string object
        sources = [
            Source(
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                score=round(number=score, ndigits=4),
            )
            for doc, score in results
        ]
        return text, sources

    return search_handbook
    #     for doc, score in results:
    #         texts.append(doc.page_content)

    #         sources.append({
    #             "sources": doc.metadata.get("sources"),
    #             "page": doc.metadata.get("page"),
    #             "score": round(score, 4),
    #         })

    #     combined_text = "\n\n".join(texts)

    #     return combined_text, sources

    # return search_handbook
