from pathlib import Path
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd

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

plt.figure(figsize=(10, 6))

# Histogram
plt.hist(df["lp"], bins=20, density=True, alpha=0.5)

# Density curve (KDE)
df["lp"].plot(kind="kde")

plt.xlabel("LPIPS")
plt.ylabel("Density")
plt.title("Histogram & Density plot of LPIPS MobileNet")
plt.grid(True)

plt.tight_layout()
plt.show()
