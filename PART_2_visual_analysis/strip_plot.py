from pathlib import Path
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Шлях до папки, де лежить сам скрипт
SCRIPT_DIR = Path(__file__).resolve().parent
html_path = SCRIPT_DIR / "report_pptx_mobilenet.html"
# Load MobileNet HTML
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

# ---- strip plot (LPIPS per document)
plt.figure(figsize=(14, 6))

docs = df["doc"].unique()
x_positions = {doc: i for i, doc in enumerate(docs)}

for doc in docs:
    lps = df[df["doc"] == doc]["lp"].values
    xs = np.random.normal(x_positions[doc], 0.04, size=len(lps))  # jitter
    plt.scatter(xs, lps, alpha=0.7)

plt.xticks(range(len(docs)), docs, rotation=45, ha="right")
plt.ylabel("LPIPS")
plt.xlabel("Document")
plt.title("LPIPS distribution per document (MobileNet, PDF→DOCX)")
plt.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
