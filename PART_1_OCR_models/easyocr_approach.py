# pip install easyocr pymupdf pillow numpy
import easyocr, fitz, io, numpy as np
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# === SETTINGS ===
PROJECT_DIR = Path(__file__).resolve().parent
PDF_FOLDER = PROJECT_DIR / "pdf_files"
OUTPUT_FOLDER = PROJECT_DIR / "results"
DPI = 300

# create folders if missing
PDF_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# initialize EasyOCR reader
reader = easyocr.Reader(['uk', 'en'], gpu=False)


def preprocess_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply adaptive threshold for better binarization
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # Denoise & sharpen
    th = cv2.medianBlur(th, 3)
    th = cv2.GaussianBlur(th, (1, 1), 0)
    return th


def page_to_pil(page, dpi=DPI):
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def process_pdf(pdf_path: Path):
    """Extract text from one PDF and save as Markdown"""
    lines = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            img = page_to_pil(page)
            proc_img = preprocess_image(img)
            results = reader.readtext(proc_img, paragraph=True, detail=0)
            lines.append(f"# Page {i}\n")
            for text in results:
                lines.append(text)
            lines.append("")

    md = "\n".join(lines)
    output_file = OUTPUT_FOLDER / f"{pdf_path.stem}.md"
    output_file.write_text(md, encoding="utf-8")
    print(f"✅ Saved: {output_file}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
    else:
        for pdf in pdf_files:
            print(f"Processing: {pdf.name}")
            process_pdf(pdf)
