#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Запуск OLMoCR-2-7B-1025 у INT4 (bitsandbytes) під 8 GB VRAM:
- Квантизація 4-bit (NF4), device_map="auto" (офлоуд на RAM за потреби)
- Рендер PDF-сторінок у PNG (base64) зі зменшеним розміром (за замовч. 1024)
- Обробка цілої папки з PDF, збереження YAML-виходу по сторінках
"""

import os
import sys
import glob
import base64
import json
import argparse
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import pypdfium2 as pdfium

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

# з olmocr беремо рендер і YAML-промпт
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt


def parse_args():
    p = argparse.ArgumentParser(
        description="Run OLMoCR-2-7B-1025 INT4 over a folder of PDFs"
    )
    p.add_argument("--input_dir", type=str, required=True, help="Папка з PDF")
    p.add_argument(
        "--output_dir", type=str, required=True, help="Куди зберігати результати"
    )
    p.add_argument(
        "--longest_dim",
        type=int,
        default=1024,
        help="Довга сторона рендеру зображення (пікс), 768..1288",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=128, help="Ліміт токенів генерації"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Температура (0.0 = детерміновано)",
    )
    p.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Які сторінки брати (напр. '1-3,5,10-12'). Якщо None — усі.",
    )
    p.add_argument(
        "--stop_on_error", action="store_true", help="Зупинятись при першій помилці"
    )
    return p.parse_args()


def parse_page_selection(pages_str, num_pages):
    """Повертає відсортований список сторінок (1-базована нумерація)"""
    if not pages_str:
        return list(range(1, num_pages + 1))
    pages = set()
    for chunk in pages_str.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = int(a), int(b)
            for i in range(a, b + 1):
                if 1 <= i <= num_pages:
                    pages.add(i)
        else:
            i = int(chunk)
            if 1 <= i <= num_pages:
                pages.add(i)
    return sorted(pages)


def load_model_int4():
    """
    Завантажуємо OLMoCR-2-7B-1025 у 4-bit (NF4) з офлоудом.
    """
    print("Loading model in 4-bit…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_name = "allenai/olmOCR-2-7B-1025"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",  # важливо: сам офлоудить між GPU/CPU
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
    )

    # Додатково: якщо є CUDA — увімкнемо tf32 для швидших матричних множень (не впливає на VRAM)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return model, processor


def run_inference_on_page(
    model,
    processor,
    image_base64,
    max_new_tokens=128,
    temperature=0.0,
):
    """
    Формуємо YAML-prompt + картинку, ганяємо через процесор і модель, повертаємо текст.
    """
    # Побудова повідомлень (YAML-інструкція + зображення)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }
    ]

    # Застосовуємо chat-template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Відкриваємо PNG із base64
    main_image = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")

    # Готуємо тензори
    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )

    # На правильний девайс (device_map="auto" вже працює в моделі; тут просто безпечне приведення)
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }

    # Генерація — економний режим (без семплінгу)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False if temperature == 0.0 else True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    # Відтинаємо префікс-промпт і декодуємо тільки нові токени
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Повертаємо перший (єдиний) результат
    return text_output[0].strip()


def get_num_pages(pdf_path):
    doc = pdfium.PdfDocument(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


def main():
    args = parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Завантажуємо модель
    model, processor = load_model_int4()
    print("Model ready.")

    pdf_paths = sorted(glob.glob(str(in_dir / "*.pdf")))
    if not pdf_paths:
        print(f"[*] У папці {in_dir} PDF не знайдено.")
        sys.exit(1)

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        pdf_out_dir = out_dir / pdf_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            num_pages = get_num_pages(pdf_path)
        except Exception as e:
            print(f"[!] Не вдалось відкрити {pdf_path.name}: {e}")
            if args.stop_on_error:
                raise
            continue

        pages_to_run = parse_page_selection(args.pages, num_pages)
        print(f"[{pdf_path.name}] сторінок: {num_pages} | до обробки: {pages_to_run}")

        for page_idx in tqdm(
            pages_to_run, desc=f"Processing {pdf_path.name}", unit="page"
        ):
            try:
                # Рендеримо у base64-PNG з контрольованим розміром
                image_b64 = render_pdf_to_base64png(
                    str(pdf_path), page_idx, target_longest_image_dim=args.longest_dim
                )

                # Запуск інференсу
                yaml_text = run_inference_on_page(
                    model=model,
                    processor=processor,
                    image_base64=image_b64,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

                # Зберігаємо результат .yaml (або .txt)
                out_fp = pdf_out_dir / f"page_{page_idx:04d}.yaml"
                with open(out_fp, "w", encoding="utf-8") as f:
                    f.write(yaml_text + "\n")

            except torch.cuda.OutOfMemoryError:
                print(
                    "\n[OOM] Закінчилась VRAM. Порада: зменшити --longest_dim (напр. 896 або 768) "
                    "або закрити процеси, що тримають GPU."
                )
                if args.stop_on_error:
                    raise
            except Exception as e:
                print(f"[!] Помилка на сторінці {page_idx}: {e}")
                if args.stop_on_error:
                    raise

    print("Готово.")


if __name__ == "__main__":
    main()
