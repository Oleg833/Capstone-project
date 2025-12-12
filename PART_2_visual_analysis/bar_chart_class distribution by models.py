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
res_df = load_report(base / "report_pptx_resnet50.html")
mob_df = load_report(base / "report_pptx_mobilenet.html")
eff_df = load_report(base / "report_pptx_efficientB3.html")

for df in (res_df, mob_df, eff_df):
    df["class"] = df.apply(lambda r: classify(r["cos"], r["lp"]), axis=1)

models = {"ResNet": res_df, "MobileNet": mob_df, "EfficientB3": eff_df}

class_order = ["OK", "Minor", "Moderate", "Severe"]
counts = {
    m: df["class"].value_counts().reindex(class_order, fill_value=0)
    for m, df in models.items()
}

plt.figure(figsize=(10, 6))

x = range(len(class_order))
colors = ["lightgreen", "lightskyblue", "orange"]  # 3 кольори для 3 моделей

plt.bar(
    [i - 0.25 for i in x],
    [counts["ResNet"][c] for c in class_order],
    width=0.25,
    label="ResNet",
    color=colors[0],
)
plt.bar(
    x,
    [counts["MobileNet"][c] for c in class_order],
    width=0.25,
    label="MobileNet",
    color=colors[1],
)
plt.bar(
    [i + 0.25 for i in x],
    [counts["EfficientB3"][c] for c in class_order],
    width=0.25,
    label="EfficientB3",
    color=colors[2],
)

plt.xticks(x, class_order)
plt.ylabel("Кількість сторінок")
# plt.title("Bar chart: class distribution by models")
plt.legend()

plt.tight_layout()
plt.show()
