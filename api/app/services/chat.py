
# from langchain.chat_models import init_chat_model
# from langchain_qdrant import QdrantVectorStore
# from pydantic import BaseModel, Field

# from core.config import (
#     get_settings,
#     get_qdrant_client,
#     get_embeddings,
# )


# # ==========================================
# # 1. RESPONSE MODELS
# # ==========================================

# class Source(BaseModel):
#     source: str
#     page: int | None = None


# class Answer(BaseModel):
#     question: str
#     answer: str
#     sources: list[Source] = Field(default_factory=list)


# # ==========================================
# # 2. RAG SERVICE
# # ==========================================

# class RAGService:

#     def __init__(self):

#         self.settings = get_settings()

#         # Qdrant connection
#         self.client = get_qdrant_client(self.settings)

#         # Embedding model
#         self.embeddings = get_embeddings(self.settings)

#         # Vector store
#         self.store = QdrantVectorStore(
#             client=self.client,
#             collection_name=self.settings.collection_name,
#             embedding=self.embeddings,
#         )

#         # Chat model
#         self.chat_model = init_chat_model(
#             model=self.settings.chat_model,
#             model_provider=self.settings.chat_provider,
#             temperature=self.settings.chat_temperature,
#         )

#     # ======================================
#     # 3. RETRIEVAL
#     # ======================================

#     def retrieve(self, question: str):

#         return self.store.similarity_search(
#             query=question,
#             k=self.settings.top_k,
#         )

#     # ======================================
#     # 4. CONTEXT
#     # ======================================

#     def prepare_context(self, documents):

#         return "\n\n".join(
#             f"Source: {doc.metadata.get('source', 'unknown')}\n"
#             f"Page: {doc.metadata.get('page', 'unknown')}\n"
#             f"Content: {doc.page_content}"
#             for doc in documents
#         )

#     # ======================================
#     # 5. LLM RESPONSE
#     # ======================================

#     def generate_answer(self, question: str, context: str):

#         prompt = f"""
# {self.settings.system_prompt}

# Answer the question using only the provided context.
# If the answer is not available in the context,
# clearly say that you don't know.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#         response = self.chat_model.invoke(prompt)

#         return response.content

#     # ======================================
#     # 6. ASK
#     # ======================================

#     def ask(self, question: str) -> Answer:

#         question = question.strip()

#         if not question:
#             raise ValueError("Question cannot be empty.")

#         # Retrieve
#         documents = self.retrieve(question)

#         if not documents:
#             return Answer(
#                 question=question,
#                 answer="I don't know based on the available handbook.",
#                 sources=[],
#             )

#         # Context
#         context = self.prepare_context(documents)

#         # Generate answer
#         answer_text = self.generate_answer(
#             question=question,
#             context=context,
#         )

#         # Sources
#         sources = [
#             Source(
#                 source=doc.metadata.get("source", "unknown"),
#                 page=doc.metadata.get("page"),
#             )
#             for doc in documents
#         ]

#         return Answer(
#             question=question,
#             answer=answer_text,
#             sources=sources,
#         )


from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field

from core.config import (
    get_settings,
    get_qdrant_client,
    get_embeddings,
)


# ==========================================
# 1. RESPONSE MODELS
# ==========================================

class Source(BaseModel):
    source: str
    page: int | None = None


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)


# ==========================================
# 2. RAG SERVICE
# ==========================================

class RagService:

    def __init__(self):

        self.settings = get_settings()

        # Qdrant connection
        self.client = get_qdrant_client(self.settings)

        # Embedding model
        self.embeddings = get_embeddings(self.settings)

        # Vector store
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding=self.embeddings,
        )

        # Chat model
        self.chat_model = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            temperature=self.settings.chat_temperature,
        )

    # ======================================
    # 3. RETRIEVAL
    # ======================================

    def retrieve(self, question: str):

        return self.store.similarity_search(
            query=question,
            k=self.settings.top_k,
        )

    # ======================================
    # 4. CONTEXT
    # ======================================

    def prepare_context(self, documents):

        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Page: {doc.metadata.get('page', 'unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in documents
        )

    # ======================================
    # 5. LLM RESPONSE
    # ======================================

    def generate_answer(self, question: str, context: str):

        prompt = f"""
{self.settings.system_prompt}

Answer the question using only the provided context.
If the answer is not available in the context,
clearly say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.chat_model.invoke(prompt)

        return response.content

    # ======================================
    # 6. ASK
    # ======================================

    def ask(self, question: str) -> Answer:

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        # Retrieve documents
        documents = self.retrieve(question)

        if not documents:
            return Answer(
                question=question,
                answer="I don't know based on the available handbook.",
                sources=[],
            )

        # Prepare context
        context = self.prepare_context(documents)

        # Generate answer
        answer_text = self.generate_answer(
            question=question,
            context=context,
        )

        # Extract sources
        sources = [
            Source(
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
            )
            for doc in documents
        ]

        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
        )

    # ======================================
    # 7. CLOSE CONNECTION
    # ======================================

    def close(self):

        if self.client:
            self.client.close()