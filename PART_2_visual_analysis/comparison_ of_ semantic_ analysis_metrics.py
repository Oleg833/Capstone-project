import matplotlib.pyplot as plt

# Data
metrics = ["Overall similarity", "Label similarity", "Numeric similarity"]
values = [0.855, 0.7452, 0.9648]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color="#CBB8EC")
plt.ylabel("Similarity score")
# plt.title("Gemini vs OLMoCR — Comparison of semantic analysis metrics")
plt.ylim(0, 1)

# plt.xticks(rotation=20)
plt.xticks()
plt.tight_layout()
plt.show()
