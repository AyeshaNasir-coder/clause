
from app.services.chat import RagService

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




