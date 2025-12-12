import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import fitz
import io
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# === SETTINGS ===
PROJECT_DIR = Path(__file__).resolve().parent
PDF_FOLDER = PROJECT_DIR / "pdf_files"
OUTPUT_FOLDER = PROJECT_DIR / "results"
DPI = 400

PDF_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


def page_to_pil(page, dpi=DPI):
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def preprocess_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )
    return th


def process_pdf(pdf_path: Path):
    lines = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            pil_img = page_to_pil(page)
            proc_img = preprocess_image(pil_img)
            text = pytesseract.image_to_string(proc_img, lang="ukr")
            lines.append(f"# Page {i}\n{text.strip()}\n")

    output_file = OUTPUT_FOLDER / f"{pdf_path.stem}.md"
    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Saved: {output_file}")


if __name__ == "__main__":
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
    else:
        for pdf in pdf_files:
            print(f"Processing: {pdf.name}")
            process_pdf(pdf)
