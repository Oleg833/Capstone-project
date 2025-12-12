import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
        page_id = 0
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue
            try:
                cos = float(tds[1].text)
                lp = float(tds[2].text)
            except:
                continue
            page_id += 1
            uid = f"{doc}|p{page_id}"
            rows.append((model_name, uid, lp))
    return pd.DataFrame(rows, columns=["model", "page_id", "lp"])


dfs = []
for model_name, filename in [
    ("ResNet", "report_pptx_resnet50.html"),
    ("MobNet", "report_pptx_mobilenet.html"),
    ("EffintB3", "report_pptx_efficientB3.html"),
]:
    df = load_report(base / filename)
    dfs.append(df)

heat_df = pd.concat(dfs).drop_duplicates()

pivot = heat_df.pivot_table(
    index="model", columns="page_id", values="lp", aggfunc="mean"
)
pivot = pivot.reindex(sorted(pivot.columns), axis=1)

# Custom colormap: low=dark blue → mid=yellow → high=red
colors = ["#00FF44", "#3C3CA3", "#FF0000"]  # dark blue → yellow → red
# colors = ["lightgreen", "blue", "orange", "red"]  # 0,1,2,3
severe_cmap = LinearSegmentedColormap.from_list("severe_map", colors)

plt.figure(figsize=(18, 4))
im = plt.imshow(pivot.values, aspect="auto", cmap=severe_cmap, vmin=0, vmax=1)

plt.yticks(range(len(pivot.index)), pivot.index)
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=6)

# Colorbar дуже близько
cbar = plt.colorbar(im, shrink=0.8, pad=0.001)
# cbar = plt.colorbar(im)
cbar.set_label("LPIPS")

# plt.title("LPIPS Heatmap by pages and models")
plt.tight_layout()
plt.show()
