# api/evals/run.py
# Run from api/:  uv run python -m evals.run
import json
from app.services.chat import RAGService

REFUSALS = ["don't know", "do not know", "not mention", "doesn't mention",
            "doesn't cover", "does not cover", "no information"]


def clean(text):
    return text.lower().replace(",", "").replace("’", "'")


def has(text, word):
    # "a|b" means a or b
    return any(option in text for option in word.split("|"))


def pct(results):
    return f"{sum(results) / len(results):.0%}" if results else "n/a"


tests = [json.loads(line) for line in open("evals/golden.jsonl") if line.strip()]
rag = RAGService()
hits, fact_found, correct, refused, leaked = [], [], [], [], []

for t in tests:
    result = rag.ask(t["question"])
    answer = clean(result.answer)

    if t["type"] == "answerable":
        # What did the librarian actually hand the writer?
        chunks = rag.store.similarity_search(t["question"], k=rag.settings.top_k)
        context = clean(" ".join(c.page_content for c in chunks))

        found = all(has(context, w) for w in t["must_include"])
        ok = all(has(answer, w) for w in t["must_include"])
        hits.append(any(s.page in t["expected_pages"] for s in result.sources))
        fact_found.append(found)
        correct.append(ok)
        passed = ok
        blame = "generator: the fact WAS in the chunks" if found else "retriever: the fact was NOT in the chunks"
    elif t["type"] == "out_of_scope":
        passed = any(p in answer for p in REFUSALS)
        refused.append(passed)
        blame = "should have said it doesn't know"
    else:
        passed = not any(has(answer, w) for w in t["must_not_include"])
        leaked.append(not passed)
        blame = "leaked staff-only information"

    print("✓" if passed else "✗", t["id"])
    if not passed:
        print("   answer:", result.answer)   # full answer, not cut off
        print("   blame: ", blame)

rag.close()

print("\nRetrieval hit rate (page):", pct(hits))
print("Fact in chunks (stricter): ", pct(fact_found))
print("Answer correctness:        ", pct(correct))
print("Refusal rate:              ", pct(refused))
print("Leak rate:                 ", pct(leaked))