import matplotlib.pyplot as plt

# Categories and approximate frequencies (based on qualitative analysis)
categories = [
    "Лексичні розбіжності\n(помилки в заголовках)",
    "Структурні помилки\n(втрати розділів/таблиць)",
    "Контекстні помилки\n(нечитання службових полів)",
    "Числові дрібні розбіжності\n(+/- 1–3 одиниці)",
]

values = [40, 30, 20, 10]  # qualitative distribution in %

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color="#CBB8EC")
plt.ylabel("Частка помилок, %")
# plt.title("Якісний аналіз помилок — Gemini vs OLMoCR")
plt.ylim(0, 50)

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
