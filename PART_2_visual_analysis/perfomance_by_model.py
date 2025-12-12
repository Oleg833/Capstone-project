import matplotlib.pyplot as plt

models = ["MobileNet", "ResNet-50", "EfficientNet-B3"]
times = [358.15, 383.37, 383.37]

colors = ["lightgreen", "lightskyblue", "gold"]

plt.figure(figsize=(8, 4))
plt.barh(models, times, color=colors)

plt.xlabel("Час обробки, сек")
# plt.title("Порівняння швидкодії моделей (Total Processing Time)")
plt.gca().invert_yaxis()

ax = plt.gca()
ax.set_xlim(0, max(times) * 1.15)  # трохи запасу справа

for idx, val in enumerate(times):
    plt.text(val + 2, idx, f"{val:.1f} s", va="center")

# for idx, val in enumerate(times):
#     plt.text(val - 5, idx, f"{val:.1f} s", va="center", ha="right")  # всередині стовпця

plt.tight_layout()
plt.show()
