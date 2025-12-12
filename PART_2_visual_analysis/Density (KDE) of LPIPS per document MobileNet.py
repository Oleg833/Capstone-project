import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

# Шлях до папки, де лежить сам скрипт
SCRIPT_DIR = Path(__file__).resolve().parent
html_path = SCRIPT_DIR / "report_pptx_mobilenet.html"

print("Шукаю файл за шляхом:", html_path)  # на всяк випадок

# Load HTML
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

# Density (KDE) LPIPS по документах
fig, ax = plt.subplots(figsize=(12, 6))

for doc, sub in df.groupby("doc"):
    # KDE коректно працює тільки якщо є хоча б два різних значення LPIPS
    if sub["lp"].nunique() > 1:
        sub["lp"].plot(kind="kde", ax=ax, label=doc)

ax.set_xlabel("LPIPS")
ax.set_ylabel("Density")
ax.set_title("Density (KDE) of LPIPS per document MobileNet")
ax.grid(True)
ax.legend(title="Document", fontsize=8)

plt.tight_layout()
plt.show()
