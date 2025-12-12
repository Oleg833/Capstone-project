import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
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
        except:
            continue
        rows.append((doc, cos, lp))

df = pd.DataFrame(rows, columns=["doc", "cos", "lp"])

# Prepare data for boxplot
docs = df["doc"].unique()
lp_data = [df[df["doc"] == doc]["lp"].values for doc in docs]

# Plot boxplot
plt.figure(figsize=(14, 6))
plt.boxplot(lp_data, labels=docs, showfliers=True)

plt.xticks(rotation=45, ha="right")
plt.ylabel("LPIPS")
plt.xlabel("Document")
plt.title("Boxplot of LPIPS per Document MobileNet")
plt.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
