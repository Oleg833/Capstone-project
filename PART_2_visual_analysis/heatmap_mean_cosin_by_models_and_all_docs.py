import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

# === Завантаження даних з трьох звітів ===
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


res_df = load_report(base / "report_pptx_resnet50.html")
mob_df = load_report(base / "report_pptx_mobilenet.html")
eff_df = load_report(base / "report_pptx_efficientB3.html")

models = {
    "ResNet": res_df,
    "MobNet": mob_df,
    "EffintB3": eff_df,
}

# === Обчислюємо середні Cosine та LPIPS по документах для кожної моделі ===
mean_rows = []
for model_name, df in models.items():
    for doc, sub in df.groupby("doc"):
        mean_cos = sub["cos"].mean()
        mean_lp = sub["lp"].mean()
        mean_rows.append((model_name, doc, mean_cos, mean_lp))

mean_df = pd.DataFrame(mean_rows, columns=["model", "doc", "mean_cos", "mean_lp"])

# Створюємо матриці для heatmap'ів
cos_pivot = mean_df.pivot(index="model", columns="doc", values="mean_cos")
lp_pivot = mean_df.pivot(index="model", columns="doc", values="mean_lp")

# === Cosine heatmap ===
fig1, ax1 = plt.subplots(figsize=(14, 5))
im1 = ax1.imshow(cos_pivot.values, aspect="auto")
ax1.set_yticks(range(len(cos_pivot.index)))
ax1.set_yticklabels(cos_pivot.index)
ax1.set_xticks(range(len(cos_pivot.columns)))
ax1.set_xticklabels(cos_pivot.columns, rotation=45, ha="right")
cbar1 = plt.colorbar(im1, ax=ax1, pad=0.003)
cbar1.set_label("Mean Cosine")
# ax1.set_title("heatmap_average_cosin_ similarity_by_models and docs")
plt.tight_layout()
plt.show()
# === LPIPS heatmap ===
# fig2, ax2 = plt.subplots(figsize=(14, 5))
# im2 = ax2.imshow(lp_pivot.values, aspect="auto")
# ax2.set_yticks(range(len(lp_pivot.index)))
# ax2.set_yticklabels(lp_pivot.index)
# ax2.set_xticks(range(len(lp_pivot.columns)))
# ax2.set_xticklabels(lp_pivot.columns, rotation=45, ha="right")
# # Colorbar дуже близько
# cbar2 = plt.colorbar(im2, ax=ax2)
# # cbar2 = plt.colorbar(im2, ax=ax2)
# cbar2.set_label("Середній LPIPS")
# ax2.set_title("LPIPS heatmap: середня візуальна відмінність по моделях і документах")
# plt.tight_layout()

# (fig1, fig2)
# fig1
