from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from app.core.config import get_embeddings, get_qdrant_client, get_settings

NO_ANSWER = "I don't know. The handbook does not provide an answer to your question. Please ask a different question or contact support for assistance."

class Source(BaseModel):
    source: str
    page: int | None = None
    score: float

class Answer(BaseModel):
    answer: str
    sources: list[Source]

class RAGService():
    def __init__(self):
        self.settings = get_settings()

        #store
        self.client = get_qdrant_client(self.settings)   #qdrant client
        if not self.client.collection_exists(self.settings.collection_name):
            raise RuntimeError(f"collection {self.settings.collection_name} not found!")
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,   #collection
            embedding=get_embeddings(self.settings),    #embed
        )

        #llm
        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            temperature=self.settings.chat_temperature,
        )

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question must be not empty")

        results = self.store.similarity_search_with_score(
            query=question,
            k=self.settings.top_k,
            score_threshold=self.settings.score_threshold
        )

        if not results:
            return Answer(answer=NO_ANSWER, sources=[])

        context = "\n\n".join(doc.page_content for doc, _ in results)
        messages = [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ]

        return Answer(
            answer = self.llm.invoke(messages).text,
            sources =[
                Source(
                    source = doc.metadata.get("source", "unknown"),
                    page = doc.metadata.get("page", None),
                    score = round(score, 4),
                )
                for doc, score in results
            ]
        )

    def close(self) -> None:
        self.client.close()


def main():
    rag_service = RAGService()
    try:
        query = input("You: ").strip()
        result = rag_service.ask(query)
    finally:
        rag_service.close()