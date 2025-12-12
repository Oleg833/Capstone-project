from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd

# Шлях до папки, де лежить сам скрипт
SCRIPT_DIR = Path(__file__).resolve().parent
path = SCRIPT_DIR / "report_pptx_mobilenet.html"

print("Шукаю файл за шляхом:", path)  # на всяк випадок

text = path.read_text(encoding="utf-8", errors="ignore")
soup = BeautifulSoup(text, "html.parser")

records = []
seen = set()

for h2 in soup.find_all("h2"):
    if "Document:" in h2.text:
        doc_id = h2.text.replace("Document:", "").strip()
        if doc_id in seen:
            continue
        seen.add(doc_id)
        table = h2.find_next("table")
        if not table:
            continue
        for tr in table.find_all("tr")[1:]:
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) == 4:
                page, cos, lp, issue = tds
                records.append(
                    [doc_id, int(page), float(cos), float(lp), issue.split()[0]]
                )

df = pd.DataFrame(records, columns=["doc_id", "page", "cosine", "lpips", "issue"])


# aggregate
def worst(issues):
    order = {"OK": 0, "Minor": 1, "Moderate": 2, "Severe": 3}
    return max(issues, key=lambda x: order.get(x, 0))


summary = df.groupby("doc_id").agg(
    mean_cosine=("cosine", "mean"),
    mean_lpips=("lpips", "mean"),
    pages=("page", "count"),
)
summary["issue"] = df.groupby("doc_id")["issue"].apply(worst)
summary["share"] = summary["pages"] / summary["pages"].sum()

summary_rounded = summary.copy()
summary_rounded["mean_cosine"] = summary["mean_cosine"].round(4)
summary_rounded["mean_lpips"] = summary["mean_lpips"].round(4)
summary_rounded["share"] = summary["share"].round(4)

summary_rounded.reset_index()

print(summary_rounded.reset_index())
df.head
