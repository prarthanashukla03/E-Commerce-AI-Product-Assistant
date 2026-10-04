from rag import generate_answer


print("====================================")
print("E-COMMERCE RAG ASSISTANT TEST")
print("====================================")


query = input("\nEnter your question: ")


answer, products = generate_answer(query)


print("\n====================================")
print("AI ANSWER")
print("====================================")

print(answer)


print("\n====================================")
print("RETRIEVED PRODUCTS")
print("====================================")

for i, product in enumerate(products):

    print(f"\n--- Product {i + 1} ---")
    print(product)