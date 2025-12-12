from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

# Шлях до папки, де лежить сам скрипт
SCRIPT_DIR = Path(__file__).resolve().parent
path = SCRIPT_DIR / "visual_report_pptx.html"

print("Шукаю файл за шляхом:", path)  # на всяк випадок
# load html
# path = "/mnt/data/visual_report_pptx.html"
with open(path, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
# find all rows
data = []
for tr in soup.find_all("tr"):
    tds = tr.find_all("td")
    if len(tds) == 4:
        page = tds[0].text.strip()
        try:
            cos = float(tds[1].text)
            lp = float(tds[2].text)
        except:
            continue
        issue_td = tds[3].text.lower()
        if "ok" in issue_td:
            cls = "OK"
        elif "minor" in issue_td:
            cls = "Minor"
        elif "moderate" in issue_td:
            cls = "Moderate"
        elif "severe" in issue_td:
            cls = "Severe"
        else:
            continue
        data.append([cls, cos, lp])

df = pd.DataFrame(data, columns=["Class", "Cosine", "LPIPS"])
summary = df.groupby("Class").agg(
    Count=("Class", "size"), MeanCosine=("Cosine", "mean"), MeanLPIPS=("LPIPS", "mean")
)
print(summary)
