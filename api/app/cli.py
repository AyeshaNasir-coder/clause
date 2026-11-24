
from app.services.agent import ClauseAgent

def main():
    clause_agent = ClauseAgent()

    try:
        query = input("You: ").strip()
        result = clause_agent.ask(question=query)
        print(f"AI: {result}")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        clause_agent.close()




if __name__ == "__main__":
    main()




