# qwen_vl_pdf_ocr_cpu.py
from pathlib import Path
import io
import torch
import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# === CONFIG ===
PROJECT_DIR = Path(__file__).resolve().parent
PDF_FOLDER = PROJECT_DIR / "pdf_files"
OUTPUT_FOLDER = PROJECT_DIR / "results"

PDF_PATH = PDF_FOLDER / "Ф1 та Ф2 завірена 1 кв 2024 ПІВОНІЯ.pdf"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DPI = 200

print("Loading Qwen2-VL model on CPU (this may take a while)...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)


def page_to_pil(page, dpi=DPI):
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def extract_text_with_qwen(img, page_num):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Read all visible text from page {page_num} and return plain text only."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[img], return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=512)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result.strip()


def pdf_to_md(pdf_path: Path) -> str:
    all_text = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            print(f"Processing page {i}/{len(doc)} ...")
            img = page_to_pil(page)
            text = extract_text_with_qwen(img, i)
            all_text.append(f"# Page {i}\n{text}\n")
    return "\n".join(all_text)


if __name__ == "__main__":
    assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}"
    md_text = pdf_to_md(PDF_PATH)
    out_path = OUTPUT_FOLDER / f"{PDF_PATH.stem}.md"
    out_path.write_text(md_text, encoding="utf-8")
    print(f"\n✅ Saved OCR-like output to: {out_path}")
