import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

# Load three reports
base = Path(__file__).resolve().parent
print("Шукаю файл за шляхом:", base)  # на всяк випадок


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


res_df = load_report(base / "report_docx_resnet50.html")
mob_df = load_report(base / "report_pptx_mobilenet.html")
eff_df = load_report(base / "report_docx_efficientB3.html")

for df in (res_df, mob_df, eff_df):
    df["class"] = df.apply(lambda r: classify(r["cos"], r["lp"]), axis=1)

models = {"ResNet": res_df, "MobileNet": mob_df, "EfficientB3": eff_df}

class_order = ["OK", "Minor", "Moderate", "Severe"]

# Count per document per class per model
records = []
for model_name, df in models.items():
    for doc, sub in df.groupby("doc"):
        counts = sub["class"].value_counts().reindex(class_order, fill_value=0)
        total = counts.sum()
        share = counts / total if total > 0 else counts
        records.append((doc, model_name, *share.values))

cols = ["doc", "model"] + class_order
data = pd.DataFrame(records, columns=cols)

# Plot: stacked bar per document (X) for each model (stack)
docs = sorted(data["doc"].unique())
x = range(len(docs))

fig, ax = plt.subplots(figsize=(14, 7))

bottom = [0] * len(docs)

colors = {"ResNet": "lightgreen", "MobileNet": "lightskyblue", "EfficientB3": "orange"}

for model_name in models.keys():
    values = []
    for d in docs:
        row = data[(data["doc"] == d) & (data["model"] == model_name)]
        if not row.empty:
            # total share of severe/moderate/minor/ok is irrelevant—
            # We need total share per model per document:
            v = row[class_order].sum(axis=1).values[0]
        else:
            v = 0
        values.append(v)
    ax.bar(x, values, bottom=bottom, label=model_name, color=colors[model_name])
    bottom = [bottom[i] + values[i] for i in range(len(values))]

ax.set_xticks(list(x))
ax.set_xticklabels(docs, rotation=45, ha="right")
ax.set_ylabel("Частка (по моделях)")
ax.set_title("Stacked bar chart: внесок моделей у аналіз кожного документа")
ax.legend(title="Моделі")

plt.tight_layout()
plt.show()
