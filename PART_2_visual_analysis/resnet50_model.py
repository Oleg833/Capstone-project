import os
import re
import shutil
import tempfile
from pathlib import Path
import ssl
import urllib.request
import certifi
import time, psutil, statistics
import fitz  # PyMuPDF
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import lpips
import numpy as np
import cv2
from PIL import Image
from docx2pdf import convert
import comtypes.client


# ==========================================================
# CONFIG
# ==========================================================
PROJECT_DIR = Path(__file__).resolve().parent

pdf_input_dir   = PROJECT_DIR / "pdf_files"
actual_input_dir = pdf_input_dir / "converted"

results_dir     = PROJECT_DIR / "results"
pdf_images_root = results_dir / "pdf_images"
images_root     = results_dir / "office_images"
temp_pdf_dir    = results_dir / "temp_pdf"

# Create missing folders automatically
for folder in [pdf_input_dir, actual_input_dir, pdf_images_root, images_root, temp_pdf_dir]:
    folder.mkdir(parents=True, exist_ok=True)

ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
RESNET50_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"


# ==========================================================
# UTILS
# ==========================================================
def clean_folders(folders):
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)


def ensure_weights(url):
    hub_dir = torch.hub.get_dir()
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    dest = os.path.join(ckpt_dir, os.path.basename(url))
    if os.path.exists(dest):
        return dest
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_ctx) as src, open(dest, "wb") as out:
        out.write(src.read())
    return dest


def page_no(fname: str) -> int:
    m = re.search(r'\d+', fname)
    return int(m.group()) if m else -1


# ==========================================================
# DOC / PPT CONVERSION HELPERS
# ==========================================================
def convert_docx_to_pdf(docx_path, output_pdf_path):
    try:
        convert(docx_path, output_pdf_path)
    except Exception as e:
        raise Exception(f"Failed to convert {docx_path} to PDF: {e}")


def pptx_to_pdf(input_path, output_path):
    powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
    powerpoint.Visible = 1
    presentation = powerpoint.Presentations.Open(input_path)
    presentation.SaveAs(output_path, 32)
    presentation.Close()
    powerpoint.Quit()


# ==========================================================
# IMAGE EXTRACTION
# ==========================================================
def pdf_to_images(pdf_path, output_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=250)
        img = pix.tobytes("png")
        img_path = os.path.join(output_img_dir, f"page_{page_num + 1}.png")
        with open(img_path, "wb") as img_file:
            img_file.write(img)
    return output_img_dir


# ==========================================================
# FEATURE EXTRACTORS
# ==========================================================
ensure_weights(ALEXNET_URL)
ensure_weights(RESNET50_URL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lpips():
    try:
        return lpips.LPIPS(net='alex').to(device)
    except Exception as err:
        print("⚠️  LPIPS pretrained weights unavailable – using random net.\n", err)


lpips_model = load_lpips()


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


model = FeatureExtractor().to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_and_preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor.to(device)


def cosine_similarity(embedding1, embedding2):
    norm1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
    norm2 = embedding2 / embedding2.norm(dim=1, keepdim=True)
    similarity = torch.mm(norm1, norm2.t())
    return similarity.item()


# ==========================================================
# IMAGE ALIGNMENT + LPIPS
# ==========================================================
def align_images(img1, img2):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return img1
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)
    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = int(len(matches) * 0.15)
    good_matches = matches[:num_good_matches]
    if len(good_matches) >= 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        h2, w2 = img2.shape[:2]
        if M is not None and M.shape == (3, 3):
            aligned_img = cv2.warpPerspective(img1, M, (w2, h2))
            return aligned_img
        else:
            return img1
    else:
        return img1


def compute_lpips_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    aligned_img1 = align_images(img1, img2)
    if aligned_img1.shape != img2.shape:
        aligned_img1 = cv2.resize(aligned_img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)
    img1_np = cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB)
    img2_np = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1_tensor = lpips.im2tensor(img1_np).to(device)
    img2_tensor = lpips.im2tensor(img2_np).to(device)
    with torch.no_grad():
        distance = lpips_model(img1_tensor, img2_tensor)
    return distance.item()


def has_issue(cos_sim: float, lpips_score: float) -> str:
    if cos_sim >= 0.9991:
        return "OK"
    if 0.9800 <= cos_sim < 0.9991:
        return "OK" if lpips_score <= 0.13 else "MINOR ISSUE"
    if 0.9700 <= cos_sim < 0.9800:
        return "MINOR ISSUE" if lpips_score <= 0.75 else "MODERATE ISSUE"
    if 0.9200 <= cos_sim < 0.9700:
        return "MODERATE ISSUE"
    return "SEVERE ISSUE"


# ==========================================================
# MAIN IMAGE COMPARISON
# ==========================================================
def autocrop_white(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    mask = 255 - thresh
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


def compare_images_multilayer(image_path1, image_path2, doc_id, page):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    img1_cropped = autocrop_white(img1)
    img2_cropped = autocrop_white(img2)
    target_h = max(img1_cropped.shape[0], img2_cropped.shape[0])
    target_w = max(img1_cropped.shape[1], img2_cropped.shape[1])
    target_size = (target_w, target_h)
    img1_resized = cv2.resize(img1_cropped, target_size, interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2_cropped, target_size, interpolation=cv2.INTER_AREA)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
        cv2.imwrite(f1.name, img1_resized)
        cv2.imwrite(f2.name, img2_resized)
        temp_path1, temp_path2 = f1.name, f2.name
    try:
        with torch.no_grad():
            emb1 = model(load_and_preprocess(temp_path1))
            emb2 = model(load_and_preprocess(temp_path2))
            cos_sim = cosine_similarity(emb1, emb2)
        lpips_score = compute_lpips_similarity(temp_path1, temp_path2)
        level = has_issue(cos_sim, lpips_score)
        return {"doc_id": doc_id, "page": page, "cosine_similarity": cos_sim, "lpips_score": lpips_score, "issue_level": level}
    finally:
        os.remove(temp_path1)
        os.remove(temp_path2)


# ==========================================================
# DOCUMENT PROCESSING
# ==========================================================
def process_pdf_document(pdf_file, pdf_input_dir, pdf_images_root):
    pdf_path = os.path.join(pdf_input_dir, pdf_file)
    base_name = os.path.splitext(pdf_file)[0]
    output_dir = os.path.join(pdf_images_root, base_name)
    pdf_to_images(pdf_path, output_dir)
    return output_dir


def process_docx_document(docx_file, docx_input_dir, docx_images_root, temp_pdf_dir):
    docx_path = os.path.join(docx_input_dir, docx_file)
    base_name = os.path.splitext(docx_file)[0]
    temp_pdf_path = os.path.join(temp_pdf_dir, base_name + ".pdf")
    convert_docx_to_pdf(docx_path, temp_pdf_path)
    output_dir = os.path.join(docx_images_root, base_name)
    pdf_to_images(temp_pdf_path, output_dir)
    return output_dir


def process_pptx_document(pptx_file, pptx_input_dir, pptx_images_root, temp_pdf_dir):
    pptx_path = os.path.join(pptx_input_dir, pptx_file)
    base_name = os.path.splitext(pptx_file)[0]
    temp_pdf_path = os.path.join(temp_pdf_dir, base_name + ".pdf")
    pptx_to_pdf(pptx_path, temp_pdf_path)
    output_dir = os.path.join(pptx_images_root, base_name)
    pdf_to_images(temp_pdf_path, output_dir)
    return output_dir


# ==========================================================
# REPORT GENERATION
# ==========================================================
def get_matching_file_pairs(folder_expected, folder_actual, expected_ext=".pdf", actual_ext=".docx"):
    expected_files = {os.path.splitext(f)[0]: f for f in os.listdir(folder_expected) if f.lower().endswith(expected_ext)}
    actual_files = {os.path.splitext(f)[0]: f for f in os.listdir(folder_actual) if f.lower().endswith(actual_ext)}
    common_bases = set(expected_files) & set(actual_files)
    return [(expected_files[base], actual_files[base], base) for base in common_bases]


def compare_image_folders_and_report(pdf_img_folder, docx_img_folder, doc_id):
    pdf_images = sorted(os.listdir(pdf_img_folder), key=page_no)
    docx_images = sorted(os.listdir(docx_img_folder), key=page_no)
    reports = []
    if len(pdf_images) != len(docx_images):
        reports.append({"doc_id": doc_id, "page": "-", "cosine_similarity": 0.0, "lpips_score": 0.0, "issue_level": "PAGE COUNT MISMATCH"})
        return reports
    for pdf_img, docx_img in zip(pdf_images, docx_images):
        page = page_no(pdf_img)
        rep = compare_images_multilayer(os.path.join(pdf_img_folder, pdf_img), os.path.join(docx_img_folder, docx_img), doc_id, page)
        reports.append(rep)
    return reports


def generate_aggregate_html_report(reports, output_path, corrupted_files=None):
    import collections
    from itertools import groupby

    ISSUE_ICONS = {
        "OK": "✅",
        "MINOR ISSUE": "🟡",
        "MODERATE ISSUE": "🟠",
        "SEVERE ISSUE": "🔴",
        "PAGE COUNT MISMATCH": "❗",
    }

    ISSUE_LABELS = {
        "OK": "OK",
        "MINOR ISSUE": "Minor",
        "MODERATE ISSUE": "Moderate",
        "SEVERE ISSUE": "Severe",
        "PAGE COUNT MISMATCH": "Page count mismatch",
    }

    doc_set = set()
    total_pages = 0
    level_counter = collections.Counter()
    docs_with_severe = set()
    docs_with_pcm = set()

    for r in reports:
        doc_set.add(r["doc_id"])
        if r["issue_level"] == "PAGE COUNT MISMATCH":
            docs_with_pcm.add(r["doc_id"])
            level_counter["PAGE COUNT MISMATCH"] += 1
        else:
            level_counter[r["issue_level"]] += 1
            total_pages += 1
            if r["issue_level"] == "SEVERE ISSUE":
                docs_with_severe.add(r["doc_id"])

    # --- Summary block ---
    summary_html = f"""
    <div style="border:1px solid #ccc; border-radius:8px; padding:12px; background:#f8f8ff; margin-bottom:25px;">
      <h1>Summary</h1>
      <ul style="line-height:1.8em;">
        <li><b>Total documents:</b> {len(doc_set)}</li>
        <li><b>Total pages checked:</b> {total_pages}</li>
        <li><b>OK:</b> {level_counter['OK']} ✅</li>
        <li><b>Minor issues:</b> {level_counter['MINOR ISSUE']} 🟡</li>
        <li><b>Moderate issues:</b> {level_counter['MODERATE ISSUE']} 🟠</li>
        <li><b>Severe issues:</b> {level_counter['SEVERE ISSUE']} 🔴</li>
        <li><b>Page count mismatches:</b> {len(docs_with_pcm)}</li>
      </ul>
    </div>
    """

    # --- Performance summary ---
    from pathlib import Path
    perf_html = ""
    perf_report_path = results_dir / f"performance_report_{Path(output_path).stem.split('_')[-1]}.txt"
    if perf_report_path.exists():
        lines = perf_report_path.read_text(encoding="utf-8").splitlines()
        clean_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("=")]
        perf_html = "<div style='border:1px solid #ccc; border-radius:8px; padding:12px; background:#f8f8ff; margin-bottom:25px;'>"
        perf_html += "<h1>Performance Summary</h1><ul style='line-height:1.8em;'>"
        for ln in clean_lines:
            if ":" in ln:
                key, val = ln.split(":", 1)
                perf_html += f"<li><b>{key.strip()}:</b> {val.strip()}</li>"
        perf_html += "</ul></div>"

    # --- Start building HTML ---
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>DL Comparison Report</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:30px;background:#fafafa;}",
        "table{border-collapse:collapse;width:80%;margin-bottom:30px;}",
        "th,td{text-align:center;border:1px solid #ddd;padding:8px;}",
        "th{background:#f2f2f2;font-weight:bold;}",
        "h2{margin-top:40px;}",
        ".OK{background:#f9fff9;}",
        ".MINOR\\ ISSUE{background:#fffbe6;}",
        ".MODERATE\\ ISSUE{background:#fff0cc;}",
        ".SEVERE\\ ISSUE{background:#ffe5e5;}",
        ".PAGE\\ COUNT\\ MISMATCH{background:#ffe0e0;}",
        "</style></head><body>",
        summary_html,
        perf_html,
    ]

    # --- Corrupted files ---
    if corrupted_files:
        html.append("<h2 style='color:#b00'>Corrupted Files</h2>")
        html.append("<table><tr><th>File</th><th>Error</th></tr>")
        for base, err in corrupted_files:
            html.append(f"<tr class='SEVERE ISSUE'><td>{base}</td><td style='color:red'>{err}</td></tr>")
        html.append("</table>")

    # --- Document-level results ---
    reports = sorted(reports, key=lambda r: (r["doc_id"], str(r["page"])))
    for doc_id, group in groupby(reports, key=lambda r: r["doc_id"]):
        html.append(f"<h2>Document: {doc_id}</h2>")
        html.append("<table><tr><th>Page</th><th>Cosine</th><th>LPIPS</th><th>Issue</th></tr>")
        for r in group:
            lvl = r["issue_level"]
            if lvl == "PAGE COUNT MISMATCH":
                html.append("<tr class='PAGE COUNT MISMATCH'><td colspan='4' style='color:red;font-weight:bold'>Page count mismatch</td></tr>")
                continue
            label = ISSUE_LABELS.get(lvl, lvl)
            icon = ISSUE_ICONS.get(lvl, "")
            html.append(
                f"<tr class='{lvl}'>"
                f"<td>{r['page']}</td>"
                f"<td>{r['cosine_similarity']:.4f}</td>"
                f"<td>{r['lpips_score']:.4f}</td>"
                f"<td>{label} {icon}</td>"
                "</tr>"
            )
        html.append("</table>")

    html.append("</body></html>")
    Path(output_path).write_text("\n".join(html), encoding="utf-8")


    for doc_id, group in groupby(reports, key=lambda r: r["doc_id"]):
        html.append(f"<h2>Document: <b>{doc_id}</b></h2>")
        html.append(
            "<table>"
            "<tr>"
            "<th style='width:15%'>Page</th>"
            "<th style='width:25%'>Cosine</th>"
            "<th style='width:25%'>LPIPS</th>"
            "<th style='width:35%'>Issue</th>"
            "</tr>"
        )
        for r in group:
            lvl = r["issue_level"]
            if lvl == "PAGE COUNT MISMATCH":
                html.append(
                    f"<tr class='PAGE COUNT MISMATCH'><td colspan='4' style='color:red;font-weight:bold'>"
                    "Page count mismatch</td></tr>"
                )
                continue
            label = ISSUE_LABELS.get(lvl, lvl)
            icon = ISSUE_ICONS.get(lvl, "")
            html.append(
                f"<tr class='{lvl}'>"
                f"<td>{r['page']}</td>"
                f"<td>{r['cosine_similarity']:.4f}</td>"
                f"<td>{r['lpips_score']:.4f}</td>"
                f"<td>{label} {icon}</td>"
                "</tr>"
            )
        html.append("</table>")

    html.append("</body></html>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


# ==========================================================
# PERFORMANCE TRACKER
# ==========================================================
def get_system_usage(samples=2, interval=0.5):
    """Return average CPU% and RAM(MB) over a few samples."""
    cpu_samples = []
    mem_samples = []
    for _ in range(samples):
        cpu_samples.append(psutil.cpu_percent(interval=interval))
        mem_samples.append(psutil.virtual_memory().used / (1024 ** 2))
    return statistics.mean(cpu_samples), statistics.mean(mem_samples)


# ==========================================================
# MAIN
# ==========================================================
def run_DL_content_validation(file_type):
    print(f"\n=== Running DL content validation for {file_type.upper()} ===")
    if file_type == "docx":
        folder_expected = pdf_input_dir
        folder_actual = os.path.join(actual_input_dir, "docx")
    elif file_type == "pptx":
        folder_expected = pdf_input_dir
        folder_actual = os.path.join(actual_input_dir, "pptx")
    else:
        raise ValueError("Unsupported file_type")

    clean_folders([pdf_images_root, images_root, temp_pdf_dir])
    common_pairs = get_matching_file_pairs(folder_expected, folder_actual, ".pdf", f".{file_type}")
    common_pairs = sorted(common_pairs, key=lambda pair: pair[0])[:115]

    all_reports, corrupted = [], []
    cpu_log, ram_log = [], []

    start_time = time.time()

    for pdf_file, other_file, base in common_pairs:
        try:
            pdf_img_folder = process_pdf_document(pdf_file, folder_expected, pdf_images_root)
            if file_type == "docx":
                doc_img_folder = process_docx_document(other_file, folder_actual, images_root, temp_pdf_dir)
            else:
                doc_img_folder = process_pptx_document(other_file, folder_actual, images_root, temp_pdf_dir)
            reports = compare_image_folders_and_report(pdf_img_folder, doc_img_folder, base)
            all_reports.extend(reports)

            cpu, ram = get_system_usage(samples=2, interval=0.3)
            cpu_log.append(cpu)
            ram_log.append(ram)

        except Exception as e:
            corrupted.append((base, str(e)))

    total_time = time.time() - start_time
    avg_cpu = round(statistics.mean(cpu_log), 2) if cpu_log else 0
    avg_ram = round(statistics.mean(ram_log), 2) if ram_log else 0

    # 🧩 Write performance report (спочатку пишемо файл)
    perf_report = results_dir / f"performance_report_{file_type}.txt"
    report_text = f"""
        === PERFORMANCE REPORT ({file_type.upper()}) ===
        Processed documents : {len(common_pairs)}
        Total time          : {total_time:.2f} sec
        Average CPU usage   : {avg_cpu:.2f} %
        Average RAM usage   : {avg_ram:.2f} MB
        ===============================
        """
    perf_report.write_text(report_text.strip(), encoding="utf-8")
    print(report_text)
    print(f"📊 Performance report saved to: {perf_report}")

    # 🧾 Generate the HTML report (тепер HTML побачить свіжий perf-файл)
    output_html = os.path.join(Path(__file__).parent, f"report_{file_type}.html")
    generate_aggregate_html_report(all_reports, output_html, corrupted)
    print(f"✅ Report generated: {output_html}")


if __name__ == "__main__":
    run_DL_content_validation("docx")
    run_DL_content_validation("pptx")
