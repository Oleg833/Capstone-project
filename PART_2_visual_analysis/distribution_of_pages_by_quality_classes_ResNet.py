from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# Шлях до папки, де лежить сам скрипт
SCRIPT_DIR = Path(__file__).resolve().parent
html_path = SCRIPT_DIR / "./report_pptx_resnet50.html"

print("Шукаю файл за шляхом:", html_path)  # на всяк випадок
# 1. Читаємо HTML-звіт MobileNet (PDF→PPTX)
with open(html_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

rows = []
for h in soup.find_all("h2"):
    doc = h.text.replace("Document:", "").strip()
    table = h.find_next("table")
    if not table:
        continue
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        try:
            cos = float(tds[1].text)
            lp = float(tds[2].text)
        except ValueError:
            continue
        rows.append((doc, cos, lp))

df = pd.DataFrame(rows, columns=["doc", "cos", "lp"])


# 2. Класифікація сторінок за правилами:
# OK: lp <= 0.11 та cos >= 0.98
# Minor: lp <= 0.22
# Moderate: lp <= 0.75
# Severe: lp > 0.75


def classify(cos, lp):
    if lp <= 0.11 and cos >= 0.98:
        return "OK"
    elif lp <= 0.22 and cos >= 0.9700:
        return "Minor"
    elif lp <= 0.75 and cos >= 0.9200:
        return "Moderate"
    else:
        return "Severe"


df["class"] = df.apply(lambda r: classify(r["cos"], r["lp"]), axis=1)

# 3. Розподіл по документах: частка сторінок кожного класу
class_order = ["OK", "Minor", "Moderate", "Severe"]

colors = {
    "OK": "lightgreen",
    "Minor": "lightskyblue",
    "Moderate": "orange",
    "Severe": "lightcoral",
}

doc_stats = (
    df.groupby(["doc", "class"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=class_order, fill_value=0)
)
doc_stats_share = doc_stats.div(doc_stats.sum(axis=1), axis=0)

# 4. Stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

bottom = None
x = range(len(doc_stats_share))
docs = doc_stats_share.index.tolist()

for cls in class_order:
    values = doc_stats_share[cls].values
    if bottom is None:
        ax.bar(x, values, label=cls, color=colors[cls])
        bottom = values
    else:
        ax.bar(x, values, bottom=bottom, label=cls, color=colors[cls])
        bottom = bottom + values

ax.set_xticks(list(x))
ax.set_xticklabels(docs, rotation=45, ha="right")
ax.set_ylabel("Частка сторінок за моделлю ResNet50")
# ax.set_xlabel("Документ")
# ax.set_title(
#     "Розподіл сторінок за класами якості для кожного документа (ResNet50, PDF→DOCX)"
# )
ax.legend(title="Клас якості")

plt.tight_layout()
plt.show()
