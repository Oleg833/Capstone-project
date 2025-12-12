import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import numpy as np

# Load reports
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


# def classify(cos, lp):
#     if cos >= 0.9991:
#         return "OK"
#     if 0.9800 <= cos < 0.9991:
#         return "OK" if lp <= 0.13 else "Minor"
#     if 0.9700 <= cos < 0.9800:
#         return "Minor" if lp <= 0.75 else "Moderate"
#     if 0.9200 <= cos < 0.9700:
#         return "Moderate"
#     return "Severe"


res_df = load_report(base / "report_pptx_resnet50.html")
mob_df = load_report(base / "report_pptx_mobilenet.html")
eff_df = load_report(base / "report_pptx_efficientB3.html")

for df in (res_df, mob_df, eff_df):
    df["class"] = df.apply(lambda r: classify(r["cos"], r["lp"]), axis=1)

models = {"ResNet": res_df, "MobNet": mob_df, "EffintB3": eff_df}

# Map classes to numbers
class_map = {"OK": 0, "Minor": 1, "Moderate": 2, "Severe": 3}

# Compute dominant class per document per model
records = []
for model_name, df in models.items():
    for doc, sub in df.groupby("doc"):
        counts = sub["class"].value_counts()
        dominant = counts.idxmax()
        records.append((doc, model_name, class_map[dominant]))

heat = pd.DataFrame(records, columns=["doc", "model", "value"])
pivot = heat.pivot(index="model", columns="doc", values="value").fillna(0)

# Custom color map list
colors = ["lightgreen", "lightskyblue", "orange", "lightcoral"]  # 0,1,2,3


# Create discrete colormap
from matplotlib.colors import ListedColormap

cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(14, 5))

im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=3, aspect="auto")

# Ticks
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right")

# Add legend
from matplotlib.patches import Patch

legend_handles = [
    Patch(color="lightgreen", label="OK"),
    Patch(color="lightskyblue", label="Minor"),
    Patch(color="orange", label="Moderate"),
    Patch(color="lightcoral", label="Severe"),
]
ax.legend(
    handles=legend_handles, title="Класи", bbox_to_anchor=(0.9, 1), loc="upper left"
)

# plt.title("Heatmap: домінуючий клас по документах і моделях")
plt.tight_layout()
plt.show()
