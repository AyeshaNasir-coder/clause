# from services.ingestion import ingest
# from services.chat import RAGService

# def main():
#     rag_service=RAGService()
#     try:
#         query = input("You:").strip()
#         result = rag_service.ask(query)
#         print(f"AI: {result.answer}")

#     except:
#         pass
#     finally:
#         rag_service.close()

# if __name__ == "__main__":
#     main() 


from services.ingestion import ingest
from services.chat import RagService

def main():
    rag_service = RagService()

    try:
        query = input("You: ").strip()
        result = rag_service.ask(query)
        print(f"AI: {result.answer}")

        print("\nSources:")
        for source in result.sources:
            print(source)

    finally:
        rag_service.close()


if __name__ == "__main__":
    main()




