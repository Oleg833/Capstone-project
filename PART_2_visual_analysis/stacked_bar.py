import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path


def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
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
            except:
                continue
            rows.append((doc, cos, lp))
    return pd.DataFrame(rows, columns=["doc", "cos", "lp"])


def classify(cos, lp):
    if lp <= 0.11 and cos >= 0.99:
        return "OK"
    elif lp <= 0.22:
        return "Minor"
    elif lp <= 0.40:
        return "Moderate"
    else:
        return "Severe"


base = Path(__file__).resolve().parent
print("Шукаю файл за шляхом:", base)  # на всяк випадок
res_df = load_report(base / "report_docx_resnet50.html")
mob_df = load_report(base / "report_pptx_mobilenet.html")
eff_df = load_report(base / "report_docx_efficientB3.html")

for df in (res_df, mob_df, eff_df):
    df["class"] = df.apply(lambda r: classify(r["cos"], r["lp"]), axis=1)

models = {"ResNet": res_df, "MobileNet": mob_df, "EfficientB3": eff_df}
class_order = ["OK", "Minor", "Moderate", "Severe"]

# Counts per class per model
counts = {
    m: df["class"].value_counts().reindex(class_order, fill_value=0)
    for m, df in models.items()
}

# Prepare stacked bar data
plt.figure(figsize=(10, 6))

x = range(len(class_order))
bottom = [0] * len(class_order)

colors = {"ResNet": "lightgreen", "MobileNet": "lightskyblue", "EfficientB3": "orange"}

for model_name, df_counts in counts.items():
    values = [df_counts[c] for c in class_order]
    plt.bar(x, values, bottom=bottom, label=model_name, color=colors[model_name])
    bottom = [bottom[i] + values[i] for i in range(len(values))]

plt.xticks(x, class_order)
plt.ylabel("Кількість сторінок")
plt.title("Stacked bar chart: вклад моделей у кожний клас якості")
plt.legend(title="Модель")

plt.tight_layout()
plt.show()
