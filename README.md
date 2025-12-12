# Capstone-project
Capstone project Neoversity
```
python run_olmocr_int4.py \
  --input_dir /path/to/pdfs \
  --output_dir /path/to/out \
  --longest_dim 1024 \
  --max_new_tokens 128 \
  --temperature 0.0
```
якщо бракує VRAM → спробуй --longest_dim 896 або 768.

щоб обробити частину сторінок: --pages "1-3,6,10-12".

```
python run_olmocr_int4.py \
  --input_dir "C:/Users/Oleh/Documents/pdfs_in" \
  --output_dir "C:/Users/Oleh/Documents/pdfs_out"
```