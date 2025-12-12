import json
import os
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

import torch

# 🔁 адаптуй під свій SDK:
# прикладовий інтерфейс; у твоїй збірці назви можуть трохи відрізнятися
# from olmo_ocr import OLMoOCR

# ---- Налаштування ----
PDF_PATH = "sample.pdf"  # шлях до твого PDF
PAGE_INDEX = 0  # яку сторінку беремо (0 = перша)
OUT_DIR = Path("olmo_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT_DIR / f"page_{PAGE_INDEX+1}.png"
JSON_PATH = OUT_DIR / f"page_{PAGE_INDEX+1}_layout.json"
ANNO_PATH = OUT_DIR / f"page_{PAGE_INDEX+1}_annotated.png"
TXT_PATH = OUT_DIR / f"page_{PAGE_INDEX+1}.txt"


# ---- 1) Рендер сторінки PDF у PNG (300 DPI) ----
def render_pdf_page(pdf_path: str, page_index: int, out_png: Path, dpi: int = 300):
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
    except IndexError:
        raise ValueError(f"PDF має {len(doc)} стор., але запитано {page_index}")
    # матриця масштабу під DPI (72 базових точки на дюйм)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # без прозорості
    pix.save(str(out_png))
    doc.close()


# ---- 2) Ініціалізація OLMoCR (GPU якщо доступний) ----
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        # MPS корисно на Mac (не актуально для RTX, але хай буде)
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# 🔁 адаптуй під свій SDK:
class DummyOLMoOCR:
    """
    Це заглушка щоб показати, на що очікує скрипт.
    Замініть на реальний клас з вашого пакету OLMoCR.
    Очікуваний протокол .read(image, return_layout=True, visualize=True, **kwargs)
    -> об'єкт із полями: text (str), layout (list[dict]), annotated_image (np.ndarray або PIL.Image)
    """

    def __init__(self, model="olmo-base", device="cpu"):
        self.model = model
        self.device = device

    def read(self, image, return_layout=True, visualize=True, **kwargs):
        # Тут має бути справжній інференс OLMoCR
        # Нижче — лише демо-структура результату.
        # image може бути шляхем, або PIL.Image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image

        # Demo: "розпізнали" умовний заголовок у центрі
        w, h = img.size
        dummy_text = "Invoice #12345\nTotal: $199.00\nThank you!"
        dummy_layout = [
            {
                "text": "Invoice #12345",
                "bbox": [int(0.2 * w), int(0.1 * h), int(0.8 * w), int(0.18 * h)],
                "type": "header",
            },
            {
                "text": "Total: $199.00",
                "bbox": [int(0.2 * w), int(0.22 * h), int(0.55 * w), int(0.28 * h)],
                "type": "value",
            },
            {
                "text": "Thank you!",
                "bbox": [int(0.2 * w), int(0.32 * h), int(0.45 * w), int(0.38 * h)],
                "type": "footer",
            },
        ]

        # “Візуалізація”: просте підмальовування рамок
        annotated = img.copy()
        draw = Image.fromarray(np.array(annotated))
        annotated = annotated.convert("RGBA")
        import PIL.ImageDraw as ImageDraw

        d = ImageDraw.Draw(annotated)
        for obj in dummy_layout:
            x1, y1, x2, y2 = obj["bbox"]
            d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
            d.text(
                (x1, max(0, y1 - 14)), obj.get("type", "text"), fill=(255, 0, 0, 255)
            )
        return type(
            "OLMoResult",
            (),
            {
                "text": dummy_text,
                "layout": dummy_layout,
                "annotated_image": annotated,  # PIL.Image expected
            },
        )


def main():
    # 1) PDF → PNG
    print(f"[1/4] Рендер PDF → PNG @300DPI: {PDF_PATH} → {PNG_PATH}")
    render_pdf_page(PDF_PATH, PAGE_INDEX, PNG_PATH, dpi=300)

    # 2) init OLMoCR
    device = pick_device()
    print(f"[2/4] Обраний пристрій: {device}")

    # 🔁 заміни DummyOLMoOCR на реальний клас з твого пакету
    # ocr = OLMoOCR(model="olmo-base-ocr", device=device)
    ocr = DummyOLMoOCR(model="olmo-base-ocr", device=device)

    # 3) інференс: текст + розмітка + візуалізація
    print(f"[3/4] Інференс OLMoCR (text + layout + visualize)")
    # у реальному SDK часто зустрічаються параметри:
    # return_layout=True, return_words=True, return_blocks=True, visualize=True, conf=True
    result = ocr.read(
        str(PNG_PATH),
        return_layout=True,
        visualize=True,
    )

    # 4) збереження результатів
    print(f"[4/4] Збереження результатів → {TXT_PATH}, {JSON_PATH}, {ANNO_PATH}")
    # текст
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write(result.text)

    # макет/розмітка
    # Рекомендований нейтральний JSON формат:
    payload = {
        "page_index": PAGE_INDEX,
        "image_path": str(PNG_PATH),
        "text": result.text,
        "blocks": result.layout,  # список елементів: {"text", "bbox":[x1,y1,x2,y2], "type":...}
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # анотоване зображення
    annotated_img = result.annotated_image
    if isinstance(annotated_img, Image.Image):
        annotated_img.save(ANNO_PATH)
    else:
        # якщо SDK повертає np.ndarray у BGR/RGB
        Image.fromarray(annotated_img).save(ANNO_PATH)

    print("✅ Готово!")
    print(f"Текст:      {TXT_PATH}")
    print(f"Розмітка:   {JSON_PATH}")
    print(f"Анотація:   {ANNO_PATH}")


if __name__ == "__main__":
    main()
