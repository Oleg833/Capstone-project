" -------------- This code is a sample as we don't have real Gemini API key -------------- "

from pathlib import Path
import json, io, os
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

PDF = Path(r"C:\path\doc_ocr.pdf")
MARKER_JSON = Path(r"C:\path\marker_out\doc_ocr.json")
OUT_DIR = Path(r"C:\path\marker_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Gemini setup ---
# set your API key in env: setx GEMINI_API_KEY "xxxxx"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def ocr_patch_with_gemini(pil_img: Image.Image) -> str:
    """Send a cropped cell image to Gemini and return plain text."""
    # Convert to JPEG bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    prompt = "Recognize the text in this table cell. Return only the text, no quotes."
    resp = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": buf.getvalue()}])
    return (resp.text or "").strip()

def crop_bbox(page, bbox, dpi=300) -> Image.Image:
    """
    Marker bboxes are typically in PDF pixel/point space (origin top-left in image space).
    We’ll render the whole page and then crop in pixel coords.
    """
    # Render page at DPI to pixels
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    page_img = Image.open(io.BytesIO(pix.tobytes("png")))

    # bbox expected as [x0, y0, x1, y1] in PDF points. Scale to pixels:
    x0, y0, x1, y1 = bbox
    crop = page_img.crop((int(x0*scale), int(y0*scale), int(x1*scale), int(y1*scale)))
    return crop

def load_marker_tables(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # Normalize blocks list
    blocks = data.get("blocks")
    if blocks is None:
        pages = data.get("pages")
        blocks = []
        for pg in (pages or []):
            blocks.extend(pg.get("blocks", []))
    # Keep only tables
    tables = [b for b in blocks if isinstance(b, dict) and "table" in str(b.get("type", "")).lower()]
    return tables

def iter_cell_bboxes(table_block):
    """
    Yields (row_idx, col_idx, bbox) for each cell.
    Marker JSON schemas vary; typical table['data'] contains cells or rows with bboxes.
    """
    d = table_block.get("data") or {}
    # Case A: explicit rows with cell dicts having 'bbox'
    rows = d.get("rows")
    if rows:
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                bbox = None
                if isinstance(cell, dict):
                    bbox = cell.get("bbox") or cell.get("box")
                if bbox and len(bbox) == 4:
                    yield r, c, bbox
        return
    # Case B: flat cells + n_rows/n_cols
    if all(k in d for k in ("cells", "n_rows", "n_cols")):
        cells = d["cells"]; nr, nc = int(d["n_rows"]), int(d["n_cols"])
        for r in range(nr):
            for c in range(nc):
                cell = cells[r*nc + c]
                bbox = None
                if isinstance(cell, dict):
                    bbox = cell.get("bbox") or cell.get("box")
                if bbox and len(bbox) == 4:
                    yield r, c, bbox

def main():
    tables = load_marker_tables(MARKER_JSON)
    if not tables:
        print("No tables found in Marker JSON.")
        return

    with fitz.open(PDF) as doc:
        for t_idx, tbl in enumerate(tables, 1):
            page_num = (tbl.get("page") or tbl.get("page_index") or 0)
            page = doc[page_num]
            # Estimate table grid size
            d = tbl.get("data") or {}
            if "rows" in d:
                nr, nc = len(d["rows"]), max((len(r) for r in d["rows"]), default=0)
            elif all(k in d for k in ("n_rows", "n_cols")):
                nr, nc = int(d["n_rows"]), int(d["n_cols"])
            else:
                continue

            grid = [["" for _ in range(nc)] for _ in range(nr)]
            for r, c, bbox in iter_cell_bboxes(tbl):
                try:
                    patch = crop_bbox(page, bbox, dpi=300)
                    text = ocr_patch_with_gemini(patch)
                except Exception as e:
                    text = ""  # fallback if API fails for a cell
                grid[r][c] = text

            # Save Markdown table
            md_path = OUT_DIR / f"{PDF.stem}_table_{t_idx}_gemini.md"
            md_lines = ["|" + "|".join(" --- " for _ in range(nc)) + "|"]  # header sep line
            # A simple headerless table; adapt if you know header rows
            for row in grid:
                md_lines.append("|" + "|".join(cell.replace("\n", " ").strip() for cell in row) + "|")
            md_path.write_text("\n".join(md_lines), encoding="utf-8")

            print(f"Saved: {md_path.name}")

if __name__ == "__main__":
    main()
