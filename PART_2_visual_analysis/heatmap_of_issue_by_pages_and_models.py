import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd

# ==========================
# Налаштування
# ==========================

base = Path(__file__).resolve().parent  # де лежить скрипт + HTML-звіти

FILES = [
    ("ResNet", "report_pptx_resnet50.html"),
    ("MobNet", "report_pptx_mobilenet.html"),
    ("EffintB3", "report_pptx_efficientB3.html"),
]

ISSUE_TO_VAL = {"OK": 0, "Minor": 1, "Moderate": 2, "Severe": 3}

# Кольори під 4 класи
COLORS = ["lightgreen", "lightskyblue", "orange", "lightcoral"]
cmap = ListedColormap(COLORS)
# Інтервали навколо значень 0,1,2,3
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)

legend_handles = [
    Patch(color="lightgreen", label="OK"),
    Patch(color="lightskyblue", label="Minor"),
    Patch(color="orange", label="Moderate"),
    Patch(color="lightcoral", label="Severe"),
]

# ==========================
# Парсинг Summary з HTML
# ==========================


def parse_summary(path: Path, model_name: str) -> dict:
    """Витягти Summary-блок зверху HTML."""
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    h1 = soup.find("h1", string="Summary")
    if not h1:
        return {"model": model_name}

    ul = h1.find_next("ul")
    items = {}

    for li in ul.find_all("li"):
        text = li.get_text(strip=True)
        # приклади:
        # "Total documents: 15"
        # "OK: 24 ✅"
        if ":" not in text:
            continue
        key, val = text.split(":", 1)
        key = key.strip()

        # Витягуємо перше число
        import re

        m = re.search(r"(\d+)", val)
        if m:
            items[key] = int(m.group(1))

    items["model"] = model_name
    return items


# ==========================
# Парсинг сторінок та Issue
# ==========================


def load_pages_issue(path: Path, model_name: str) -> pd.DataFrame:
    """
    Витягає:
      model, document, page (№), page_id = "doc|pX", issue ("OK"/"Minor"/...)
    Прибирає дублікати по (model, document, page), бо в HTML блоки повторюються.
    """
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    rows = []
    for h2 in soup.find_all("h2"):
        text = h2.get_text(strip=True)
        if not text.startswith("Document:"):
            continue

        doc = text.replace("Document:", "").strip()
        table = h2.find_next("table")
        if not table:
            continue

        page_id = 0
        for tr in table.find_all("tr")[1:]:  # пропускаємо заголовок
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue

            page_id += 1
            issue_cell = tds[3].get_text(strip=True)  # типу "Minor 🟡"
            issue = issue_cell.split()[0]  # беремо тільки слово
            uid = f"{doc}|p{page_id}"
            rows.append((model_name, doc, page_id, uid, issue))

    df = pd.DataFrame(rows, columns=["model", "document", "page", "page_id", "issue"])
    # Видаляємо дублікати
    df = df.drop_duplicates(subset=["model", "document", "page"])
    return df


# ==========================
# Читання всіх моделей
# ==========================

all_summaries = []
all_pages = []

for model_name, filename in FILES:
    path = base / filename
    print(f"Читаю файл: {path}")

    # Summary
    summary = parse_summary(path, model_name)
    all_summaries.append(summary)

    # Сторінки
    pages_df = load_pages_issue(path, model_name)
    all_pages.append(pages_df)

# Зведена таблиця Summary по моделях (для контролю)
summary_df = pd.DataFrame(all_summaries)
print("\n=== SUMMARY BY MODEL ===")
print(summary_df.to_string(index=False))

# ==========================
# Підготовка даних для heatmap
# ==========================

df_all = pd.concat(all_pages, ignore_index=True)

# Перетворюємо класи в числа 0..3
df_all["issue_val"] = df_all["issue"].map(ISSUE_TO_VAL)

# Порядок сторінок та моделей
order_pages = sorted(df_all["page_id"].unique())
order_models = ["ResNet", "MobNet", "EffintB3"]

pivot = df_all.pivot_table(
    index="page_id",  # рядки – сторінки ("doc|pX")
    columns="model",  # стовпці – моделі
    values="issue_val",
    aggfunc="first",
).reindex(index=order_pages, columns=order_models)

# ==========================
# Побудова heatmap (повернутий на 90°)
# ==========================

plt.figure(figsize=(10, 4))  # широка, бо сторінки по X

im = plt.imshow(
    pivot.values.T,  # ТРАНСПОЗИЦІЯ: моделі по Y, сторінки по X
    aspect="auto",
    cmap=cmap,
    norm=norm,
)

# Вісь Y — моделі
plt.yticks(
    range(len(pivot.columns)), pivot.columns, fontsize=12, rotation=90, va="center"
)

# Вісь X — сторінки
plt.xticks(range(len(pivot.index)), pivot.index, fontsize=8, rotation=90)

# Colorbar
# cbar = plt.colorbar(
#     im,
#     shrink=0.8,
#     pad=0.001,
#     boundaries=bounds,
#     ticks=[0, 1, 2, 3],
# )
# cbar.set_ticklabels(["OK", "Minor", "Moderate", "Severe"])
# cbar.set_label("Issue class")

# Легенда
plt.legend(handles=legend_handles, loc="upper left")

# plt.title("Heatmap of issue classes by model × pages")
plt.tight_layout()
plt.show()
