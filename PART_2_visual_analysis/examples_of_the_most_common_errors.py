import matplotlib.pyplot as plt

# Examples of specific error cases
examples = [
    "«перші вартості» → «первісна вартість»",
    "«групи витраття» → «групи вибуття»",
    "Втрата службових полів (КВЕД, КОПФГ)",
    "Неповне зчитування рядка таблиці",
    "Помилка чисел: -34373 → -34375",
]

freq = [30, 25, 15, 20, 10]  # relative weights of examples

plt.figure(figsize=(12, 6))
plt.barh(examples, freq, color="#CBB8EC")  # фіолетовий
plt.xlabel("Відносна частота помилок, %")
# plt.title("Приклади найтиповіших помилок — Gemini vs OLMoCR")
plt.xlim(0, 35)
plt.tight_layout()
plt.show()
