#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import re, unicodedata, html as html_lib, webbrowser, math, difflib, hashlib
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR.parent

GT_DIR = BASE_DIR / "PART_1_OCR_models" / "MD_gemini"
RES_DIR = BASE_DIR / "PART_1_OCR_models" / "MD_olmocr"
OUT_DIR = BASE_DIR / "PART_1_OCR_models" / "results"

# Matching
RECURSIVE_SEARCH = False
FUZZY_MIN_RATIO = 0.82  # difflib ratio threshold for fallback pairing
FUZZY_TOKEN_MIN_JACC = 0.55  # token Jaccard threshold for fallback pairing

# Similarity thresholds
LABEL_OK_THRESH = 0.80
NUM_REL_TOL = 0.02
CROSS_CODE_SIM_THRESH = 0.80
LABEL_ONLY_SIM_THRESH = 0.60
# ====================================================================
EPS = 1e-12  # epsilon used to prevent division-by-zero


# ---------- utils ----------
def read_text(p: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return p.read_text(errors="ignore")


def strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"</?b>|</?i>|</?strong>|</?em>", "", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return s


# Normalize look-alike Roman letters (Cyrillic ? Latin) for numeral tokens only
_ROMAN_TRANSLATE = str.maketrans(
    {
        "?": "i",
        "?": "i",
        "?": "x",
        "?": "x",
        "?": "c",
        "?": "c",
        "?": "v",
        "?": "v",
        "?": "m",
        "?": "m",
        "?": "d",
        "?": "d",
        "?": "l",
        "?": "l",
    }
)
_ROMAN_TOKEN_RE = re.compile(r"\b[ivxlcdm???????]+\b", re.I)
_LEADING_SECTION = re.compile(
    r"^\s*((?:[ivxlcdm???????]+|[0-9]+))\.?\s+", re.I
)  # strip leading I./??./1. etc.
_CYR_EQUIV = str.maketrans(
    {
        # Ukrainian -> Russian equivalents
        "?": "?",
        "?": "?",
        "?": "?",
        "?": "?",
        "?": "?",
    }
)


def harmonize_roman_tokens(text: str) -> str:
    def _repl(m: re.Match) -> str:
        return m.group(0).translate(_ROMAN_TRANSLATE)

    return _ROMAN_TOKEN_RE.sub(_repl, text)


def normalize_text(s: str) -> str:
    s = strip_html(s)
    s = unicodedata.normalize("NFKC", s).casefold()
    s = harmonize_roman_tokens(s)
    s = _LEADING_SECTION.sub("", s)  # drop leading section numerals ("i.", "1.", "іі.")
    s = re.sub(r"[^\w\s]+", " ", s)

    # compress whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_alignment_row(cells: List[str]) -> bool:
    return all(
        re.fullmatch(r"\s*:?-{3,}:?\s*", (c or "")) for c in cells if cells is not None
    )


def is_blank_row(row):
    return (not row) or all(((cell or "").strip() == "") for cell in row)


def split_markdown_tables(text: str) -> List[List[List[str]]]:
    lines = text.splitlines()
    tables, cur = [], []
    for ln in lines:
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            row_str = ln.strip()[1:-1]
            cells = [c.strip() for c in row_str.split("|")]
            cur.append(cells)
        else:
            if cur:
                if len(cur) >= 2:
                    tables.append(cur)
                cur = []
    if cur and len(cur) >= 2:
        tables.append(cur)
    cleaned = []
    for tbl in tables:
        cleaned.append(
            [tbl[0]] + tbl[2:] if len(tbl) >= 2 and is_alignment_row(tbl[1]) else tbl
        )
    return cleaned


# ---------- number / label detection ----------
_LETTERS_RE = re.compile(r"[A-Za-zА-Яа-яЁёЇїІіЄєҐґ]")


def cell_has_letters(s: str) -> bool:
    return bool(_LETTERS_RE.search(s))


def parse_number(cell: str) -> Optional[int]:
    """Parse only *pure* numeric cells (allow spaces, thin spaces, commas, dots, parentheses). Ignore if any letters present."""
    if cell is None:
        return None
    s = strip_html(cell).replace("\u00a0", " ").replace("\u202f", " ").strip()
    if s in {"-", "—", "–"}:
        return None
    if cell_has_letters(s):  # mixed '1 Власний' must be treated as label, not a number
        return None
    neg = s.startswith("(") and s.endswith(")")
    digits = re.sub(r"[^\d]", "", s)
    if digits == "":
        return None
    try:
        val = int(digits)
        return -val if neg else val
    except ValueError:
        return None


def cell_is_code(cell: str) -> bool:
    if cell is None:
        return False
    s = strip_html(cell).strip()
    return bool(re.fullmatch(r"\d{3,4}", s))


def extract_code_label_numbers_from_row(
    row: List[str],
) -> Tuple[Optional[str], str, Optional[int], Optional[int]]:
    """Return (code, label, start, end). Label must have letters; numbers only from pure numeric cells."""
    if not row:
        return None, "", None, None
    code_idx = None
    for i, c in enumerate(row):
        if cell_is_code(c):
            code_idx = i
            break
    code = strip_html(row[code_idx]).strip() if code_idx is not None else None

    label = ""
    if code_idx is not None:
        for c in reversed(row[:code_idx]):
            txt = strip_html(c or "").strip()
            if txt and cell_has_letters(txt):
                label = txt
                break
        if not label:
            for c in row[code_idx + 1 :]:
                txt = strip_html(c or "").strip()
                if txt and cell_has_letters(txt) and parse_number(c) is None:
                    label = txt
                    break
    else:
        for c in row:
            txt = strip_html(c or "").strip()
            if txt and cell_has_letters(txt):
                label = txt
                break

    # numbers: only pure numeric cells
    nums = []
    search = row[code_idx + 1 :] if code_idx is not None else row
    for c in search:
        n = parse_number(c)
        if n is not None:
            nums.append(n)
        if len(nums) >= 2:
            break
    start = nums[0] if len(nums) >= 1 else None
    end = nums[1] if len(nums) >= 2 else None
    return code, label, start, end


def extract_code_map_and_label_rows(
    tables: List[List[List[str]]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """code_map: code -> {label,start,end}; label_rows_nocode: rows with NO code (label,start,end,norm)."""
    code_map, label_rows_nocode = {}, []
    for tbl in tables:
        for row in tbl:
            if is_blank_row(row):
                continue
            code, label, start, end = extract_code_label_numbers_from_row(row)
            if code:
                if code not in code_map:
                    code_map[code] = {"label": label, "start": start, "end": end}
                else:
                    prev = code_map[code]
                    prev_n = (prev.get("start") is not None) + (
                        prev.get("end") is not None
                    )
                    curr_n = (start is not None) + (end is not None)
                    if curr_n > prev_n:
                        code_map[code] = {"label": label, "start": start, "end": end}
            else:
                norm = normalize_text(label)
                if norm:
                    label_rows_nocode.append(
                        {"label": label, "start": start, "end": end, "norm": norm}
                    )
    return code_map, label_rows_nocode


# ---------- similarity ----------
def jaccard_char_sim(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    sa = set(na.split())
    sb = set(nb.split())
    jacc = len(sa & sb) / (len(sa | sb) or 1)
    ca = set(na)
    cb = set(nb)
    char_overlap = len(ca & cb) / (len(ca | cb) or 1)
    return round(0.5 * jacc + 0.5 * char_overlap, 4)


def tokenize_meaningful(s: str) -> set[str]:
    return {t for t in normalize_text(s).split() if len(t) >= 3}


def token_jaccard(a: str, b: str) -> float:
    A, B = tokenize_meaningful(a), tokenize_meaningful(b)
    if not A and not B:
        return 1.0
    return len(A & B) / (len(A | B) or 1)


def rel_error(a: Optional[int], b: Optional[int]) -> Optional[float]:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b), EPS)
    return abs(a - b) / denom


def num_similarity(
    gs: Optional[int], rs: Optional[int], ge: Optional[int], re_: Optional[int]
) -> Optional[float]:
    s_rel = rel_error(gs, rs)
    e_rel = rel_error(ge, re_)
    s_sim = None if s_rel is None else max(0.0, 1.0 - min(1.0, s_rel))
    e_sim = None if e_rel is None else max(0.0, 1.0 - min(1.0, e_rel))
    sims = [v for v in (s_sim, e_sim) if v is not None]
    if len(sims) == 0:
        both_missing = gs is None and rs is None and ge is None and re_ is None
        return 1.0 if both_missing else None
    return round(float(np.mean(sims)), 4)


def num_similarity_with_codes(gs, rs, ge, re_, gt_has_code: bool, res_has_code: bool):
    """
    Numbers logic:
      - Both sides have at least one number  -> compute normally
      - Exactly ONE side has numbers         -> 0.0 (mismatch)
      - Neither side has numbers:
          * neither side has a code         -> 1.0 (pure headings)
          * both sides have a code          -> None (exclude from numeric avg)
          * one side has a code, other not  -> 0.0 (expected numbers on code-side)
    """
    have_gt = (gs is not None) or (ge is not None)
    have_res = (rs is not None) or (re_ is not None)

    if have_gt and have_res:
        return num_similarity(gs, rs, ge, re_)

    if have_gt != have_res:
        return 0.0

    # neither side has numbers
    if not gt_has_code and not res_has_code:
        return 1.0
    if gt_has_code and res_has_code:
        return None  # code-only on both sides -> exclude
    return 0.0  # code on one side only -> penalize


# ---------- formatting ----------
def format_numbers_with_code(
    code: Optional[str], start: Optional[int], end: Optional[int], side_has_code: bool
) -> str:
    if not side_has_code:
        return ""
    parts = [code] if code else []
    if start is not None:
        parts.append(str(start))
    if end is not None:
        parts.append(str(end))
    return " ".join(parts)


def format_numbers_no_code(start: Optional[int], end: Optional[int]) -> str:
    parts = []
    if start is not None:
        parts.append(str(start))
    if end is not None:
        parts.append(str(end))
    return " ".join(parts)


# ---------- label-only matcher (no-code on both sides) ----------
def match_label_only_rows(
    gt_rows: List[Dict[str, Any]], res_rows: List[Dict[str, Any]], sim_thresh: float
) -> List[Tuple[int, int, float]]:
    CAND_TJACCARD = 0.50
    CAND_SHARED = 2
    scored = []
    for i, g in enumerate(gt_rows):
        for j, r in enumerate(res_rows):
            sim = jaccard_char_sim(g["label"], r["label"])
            tj = token_jaccard(g["label"], r["label"])
            shared = len(
                tokenize_meaningful(g["label"]) & tokenize_meaningful(r["label"])
            )
            if (sim >= sim_thresh) or (tj >= CAND_TJACCARD) or (shared >= CAND_SHARED):
                scored.append((i, j, sim))
    scored.sort(key=lambda t: t[2], reverse=True)
    matched_gt, matched_res, out = set(), set(), []
    for i, j, sim in scored:
        if i in matched_gt or j in matched_res:
            continue
        matched_gt.add(i)
        matched_res.add(j)
        out.append((i, j, sim))
    return out


def _both_missing(gs, ge, rs, re_):
    return gs is None and ge is None and rs is None and re_ is None


# ---------- build dataframe with all pairings ----------
def build_dataframe(gt_path: Path, res_path: Path) -> pd.DataFrame:
    gt_tables = split_markdown_tables(read_text(gt_path))
    res_tables = split_markdown_tables(read_text(res_path))

    gt_map, gt_nocode = extract_code_map_and_label_rows(gt_tables)
    res_map, res_nocode = extract_code_map_and_label_rows(res_tables)

    rows = []
    used_gt_codes, used_res_codes = set(), set()

    # 1) Same-code rows
    for code in sorted(set(gt_map) & set(res_map), key=lambda x: (len(x), x)):
        g, r = gt_map[code], res_map[code]
        glabel, rs_label = g["label"], r["label"]
        gs, ge, rs, re_ = g["start"], g["end"], r["start"], r["end"]
        lab_sim = jaccard_char_sim(glabel, rs_label)
        n_sim = num_similarity_with_codes(gs, rs, ge, re_, True, True)
        if _both_missing(gs, ge, rs, re_):
            n_sim = 1.0
        s_rel, e_rel = rel_error(gs, rs), rel_error(ge, re_)
        rows.append(
            {
                "gt_label": glabel,
                "res_label": rs_label,
                "gt_numbers": format_numbers_with_code(code, gs, ge, True),
                "res_numbers": format_numbers_with_code(code, rs, re_, True),
                "label_sim": round(lab_sim, 4),
                "num_sim": n_sim,
                "overall_sim": round(
                    float(
                        np.nanmean([lab_sim, n_sim if n_sim is not None else np.nan])
                    ),
                    4,
                ),
                "start_rel_diff_%": None if s_rel is None else round(100 * s_rel, 4),
                "end_rel_diff_%": None if e_rel is None else round(100 * e_rel, 4),
            }
        )
        used_gt_codes.add(code)
        used_res_codes.add(code)

    # 2) Cross-code pairing by label (code↔code, different codes)
    rem_gt_codes = [c for c in gt_map.keys() - used_gt_codes]
    rem_res_codes = [c for c in res_map.keys() - used_res_codes]
    matched_res_codes = set()
    for cg in rem_gt_codes:
        g = gt_map[cg]
        glabel, gs, ge = g["label"], g["start"], g["end"]
        best_code, best_sim = None, -1.0
        for cr in rem_res_codes:
            if cr in matched_res_codes:
                continue
            sim = jaccard_char_sim(glabel, res_map[cr]["label"])
            if sim > best_sim:
                best_code, best_sim = cr, sim
        if best_code is not None and best_sim >= CROSS_CODE_SIM_THRESH:
            r = res_map[best_code]
            rs, re_ = r["start"], r["end"]
            lab_sim = best_sim
            n_sim = num_similarity_with_codes(gs, rs, ge, re_, True, True)
            s_rel, e_rel = rel_error(gs, rs), rel_error(ge, re_)
            rows.append(
                {
                    "gt_label": glabel,
                    "res_label": r["label"],
                    "gt_numbers": format_numbers_with_code(cg, gs, ge, True),
                    "res_numbers": format_numbers_with_code(best_code, rs, re_, True),
                    "label_sim": round(lab_sim, 4),
                    "num_sim": n_sim,
                    "overall_sim": round(
                        float(
                            np.nanmean(
                                [lab_sim, n_sim if n_sim is not None else np.nan]
                            )
                        ),
                        4,
                    ),
                    "start_rel_diff_%": (
                        None if s_rel is None else round(100 * s_rel, 4)
                    ),
                    "end_rel_diff_%": None if e_rel is None else round(100 * e_rel, 4),
                }
            )
            matched_res_codes.add(best_code)
            used_gt_codes.add(cg)
    used_res_codes |= matched_res_codes

    # 3) Mixed matching (code↔no-code)
    #    a) GT code ↔ RES no-code
    matched_res_nc = set()
    for cg in [c for c in gt_map.keys() - used_gt_codes]:
        g = gt_map[cg]
        glabel, gs, ge = g["label"], g["start"], g["end"]
        best_j, best_sim = None, -1.0
        for j, r in enumerate(res_nocode):
            if j in matched_res_nc:
                continue
            sim = jaccard_char_sim(glabel, r["label"])
            tj = token_jaccard(glabel, r["label"])
            shared = len(tokenize_meaningful(glabel) & tokenize_meaningful(r["label"]))
            if sim > best_sim and (
                sim >= CROSS_CODE_SIM_THRESH or tj >= 0.50 or shared >= 2
            ):
                best_j, best_sim = j, sim
        if best_j is not None:
            r = res_nocode[best_j]
            rs, re_ = r["start"], r["end"]
            lab_sim = best_sim
            n_sim = num_similarity_with_codes(gs, rs, ge, re_, True, False)
            s_rel, e_rel = rel_error(gs, rs), rel_error(ge, re_)
            rows.append(
                {
                    "gt_label": glabel,
                    "res_label": r["label"],
                    "gt_numbers": format_numbers_with_code(cg, gs, ge, True),
                    "res_numbers": format_numbers_no_code(rs, re_),
                    "label_sim": round(lab_sim, 4),
                    "num_sim": n_sim,
                    "overall_sim": round(
                        float(
                            np.nanmean(
                                [lab_sim, n_sim if n_sim is not None else np.nan]
                            )
                        ),
                        4,
                    ),
                    "start_rel_diff_%": (
                        None if s_rel is None else round(100 * s_rel, 4)
                    ),
                    "end_rel_diff_%": None if e_rel is None else round(100 * e_rel, 4),
                }
            )
            matched_res_nc.add(best_j)
            used_gt_codes.add(cg)

    #    b) RES code ↔ GT no-code
    matched_gt_nc = set()
    for cr in [c for c in res_map.keys() - used_res_codes]:
        r = res_map[cr]
        rs, re_ = r["start"], r["end"]
        rlabel = r["label"]
        best_i, best_sim = None, -1.0
        for i, g in enumerate(gt_nocode):
            if i in matched_gt_nc:
                continue
            sim = jaccard_char_sim(g["label"], rlabel)
            tj = token_jaccard(g["label"], rlabel)
            shared = len(tokenize_meaningful(g["label"]) & tokenize_meaningful(rlabel))
            if sim > best_sim and (
                sim >= CROSS_CODE_SIM_THRESH or tj >= 0.50 or shared >= 2
            ):
                best_i, best_sim = i, sim
        if best_i is not None:
            g = gt_nocode[best_i]
            gs, ge = g["start"], g["end"]
            lab_sim = best_sim
            n_sim = num_similarity_with_codes(gs, rs, ge, re_, False, True)
            s_rel, e_rel = rel_error(gs, rs), rel_error(ge, re_)
            rows.append(
                {
                    "gt_label": g["label"],
                    "res_label": rlabel,
                    "gt_numbers": format_numbers_no_code(gs, ge),
                    "res_numbers": format_numbers_with_code(cr, rs, re_, True),
                    "label_sim": round(lab_sim, 4),
                    "num_sim": n_sim,
                    "overall_sim": round(
                        float(
                            np.nanmean(
                                [lab_sim, n_sim if n_sim is not None else np.nan]
                            )
                        ),
                        4,
                    ),
                    "start_rel_diff_%": (
                        None if s_rel is None else round(100 * s_rel, 4)
                    ),
                    "end_rel_diff_%": None if e_rel is None else round(100 * e_rel, 4),
                }
            )
            matched_gt_nc.add(best_i)
            used_res_codes.add(cr)

    # 4) One-sided leftovers WITH CODE
    for cg in gt_map.keys() - used_gt_codes:
        g = gt_map[cg]
        glabel = g["label"]
        gs, ge = g["start"], g["end"]
        rows.append(
            {
                "gt_label": glabel,
                "res_label": "",
                "gt_numbers": format_numbers_with_code(cg, gs, ge, True),
                "res_numbers": "",
                "label_sim": np.nan,
                "num_sim": None,
                "overall_sim": None,
                "start_rel_diff_%": None,
                "end_rel_diff_%": None,
            }
        )
    for cr in res_map.keys() - used_res_codes:
        r = res_map[cr]
        rs, re_ = r["start"], r["end"]
        rows.append(
            {
                "gt_label": "",
                "res_label": r["label"],
                "gt_numbers": "",
                "res_numbers": format_numbers_with_code(cr, rs, re_, True),
                "label_sim": np.nan,
                "num_sim": None,
                "overall_sim": None,
                "start_rel_diff_%": None,
                "end_rel_diff_%": None,
            }
        )

    # 5) No-code↔no-code label-only matching
    matches = match_label_only_rows(
        gt_nocode, res_nocode, sim_thresh=LABEL_ONLY_SIM_THRESH
    )
    matched_gt_nc2 = {i for (i, _, _) in matches} | matched_gt_nc
    matched_res_nc2 = {j for (_, j, _) in matches} | matched_res_nc
    for i, j, sim in matches:
        g, r = gt_nocode[i], res_nocode[j]
        gs, ge, rs, re_ = g["start"], g["end"], r["start"], r["end"]
        n_sim = num_similarity_with_codes(gs, rs, ge, re_, False, False)
        s_rel, e_rel = rel_error(gs, rs), rel_error(ge, re_)
        rows.append(
            {
                "gt_label": g["label"],
                "res_label": r["label"],
                "gt_numbers": format_numbers_no_code(gs, ge),
                "res_numbers": format_numbers_no_code(rs, re_),
                "label_sim": round(sim, 4),
                "num_sim": n_sim,
                "overall_sim": round(
                    float(np.nanmean([sim, n_sim if n_sim is not None else np.nan])), 4
                ),
                "start_rel_diff_%": None if s_rel is None else round(100 * s_rel, 4),
                "end_rel_diff_%": None if e_rel is None else round(100 * e_rel, 4),
            }
        )

    # 6) Standalone no-code leftovers (unmatched)
    for idx, g in enumerate(gt_nocode):
        if idx in matched_gt_nc2:
            continue
        rows.append(
            {
                "gt_label": g["label"],
                "res_label": "",
                "gt_numbers": format_numbers_no_code(g["start"], g["end"]),
                "res_numbers": "",
                "label_sim": np.nan,
                "num_sim": None,
                "overall_sim": None,
                "start_rel_diff_%": None,
                "end_rel_diff_%": None,
            }
        )
    for idx, r in enumerate(res_nocode):
        if idx in matched_res_nc2:
            continue
        rows.append(
            {
                "gt_label": "",
                "res_label": r["label"],
                "gt_numbers": "",
                "res_numbers": format_numbers_no_code(r["start"], r["end"]),
                "label_sim": np.nan,
                "num_sim": None,
                "overall_sim": None,
                "start_rel_diff_%": None,
                "end_rel_diff_%": None,
            }
        )

    df = pd.DataFrame(rows)
    # sort: paired first (both labels present), then one-sided
    df["__both__"] = (df["gt_label"].str.len().fillna(0) > 0) & (
        df["res_label"].str.len().fillna(0) > 0
    )
    df = df.sort_values(
        by=["__both__", "gt_label", "res_label"], ascending=[False, True, True]
    ).drop(columns="__both__")
    front = [
        "gt_label",
        "res_label",
        "gt_numbers",
        "res_numbers",
        "label_sim",
        "num_sim",
        "overall_sim",
    ]
    return df[front + [c for c in df.columns if c not in front]]


# ---------- HTML ----------
def write_html(df: pd.DataFrame, out_html: Path, gt_path: Path, res_path: Path):
    avg_label = (
        float(df["label_sim"].dropna().mean())
        if df["label_sim"].notna().any()
        else np.nan
    )
    avg_num = (
        float(df["num_sim"].dropna().mean()) if df["num_sim"].notna().any() else np.nan
    )
    avg_over = (
        float(df["overall_sim"].dropna().mean())
        if df["overall_sim"].notna().any()
        else np.nan
    )

    NUM_DISPLAY_FILL = "mirror_label"

    def esc(s: Any) -> str:
        if s is None or (isinstance(s, float) and (np.isnan(s) or math.isinf(s))):
            return ""
        return html_lib.escape(str(s))

    def fmt_num(v: Optional[float]) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v) or math.isinf(v))):
            return "—"
        return f"{v:.4f}"

    def filled_num_display(num_sim_val, label_sim_val):
        is_missing = (num_sim_val is None) or (
            isinstance(num_sim_val, float) and np.isnan(num_sim_val)
        )
        if not is_missing:
            return num_sim_val, False
        if NUM_DISPLAY_FILL == "mirror_label":
            return (
                label_sim_val
                if not (
                    label_sim_val is None
                    or (isinstance(label_sim_val, float) and np.isnan(label_sim_val))
                )
                else 0.0
            ), True
        if NUM_DISPLAY_FILL == "one":
            return 1.0, True
        if NUM_DISPLAY_FILL == "zero":
            return 0.0, True
        return 0.0, True  # fallback

    def sim_bg_style(v: Optional[float]) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        v = max(0.0, min(1.0, float(v)))
        hue = int(120 * v)  # 0..120 (red..green)
        return f"background-color:hsl({hue},70%,88%);"

    html_rows = []
    for _, row in df.iterrows():
        label_val = row["label_sim"]
        num_val_display, filled = filled_num_display(row["num_sim"], row["label_sim"])
        overall_val = row["overall_sim"]
        badge = (
            "<span class='badge' title='display fill; not used in averages'>•</span>"
            if filled
            else ""
        )
        html_rows.append(
            "<tr>"
            f"<td class='label'>{esc(row['gt_label'])}</td>"
            f"<td class='label'>{esc(row['res_label'])}</td>"
            f"<td class='nums'>{esc(row['gt_numbers'])}</td>"
            f"<td class='nums'>{esc(row['res_numbers'])}</td>"
            f"<td class='sim' style='{sim_bg_style(label_val)}'>{fmt_num(label_val)}</td>"
            f"<td class='sim' style='{sim_bg_style(num_val_display)}'>{fmt_num(num_val_display)}{badge}</td>"
            f"<td class='sim strong' style='{sim_bg_style(overall_val)}'>{fmt_num(overall_val)}</td>"
            "</tr>"
        )

    html_header = (
        "<tr>"
        "<th>Gemini label</th>"
        "<th>OLMOCR label</th>"
        "<th>Gemini numbers</th>"
        "<th>OLMOCR numbers</th>"
        "<th>Label sim</th>"
        "<th>Number sim</th>"
        "<th>Overall sim</th>"
        "</tr>"
    )

    summary_html = f"""
    <div class="summary-cards">
      <div class="card">
        <div class="card-title">Average overall similarity</div>
        <div class="card-value">{fmt_num(avg_over)}</div>
      </div>
      <div class="card">
        <div class="card-title">Average label similarity</div>
        <div class="card-value">{fmt_num(avg_label)}</div>
      </div>
      <div class="card">
        <div class="card-title">Average numeric similarity</div>
        <div class="card-value">{fmt_num(avg_num)}</div>
      </div>
    </div>
    """

    title = "Markdown Comparison Report — OLMOCR vs Gemini"
    subtitle = f"{gt_path.name}  ↔  {res_path.name}"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{html_lib.escape(title)}</title>
<style>
  :root {{
    --bg: #0b1020;
    --fg: #111;
    --muted: #666;
    --card: #fff;
    --border: #e6e6e6;
  }}
  body {{
    font-family: Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    margin: 0; color: var(--fg); background: #f7f8fb;
  }}
  .topbar {{
    background: var(--bg); color: #fff; padding: 18px 24px;
    font-size: 18px; font-weight: 600;
  }}
  .subtitle {{ font-size: 13px; color: #c9d2ff; opacity: .9; margin-top: 4px; }}
  .container {{ padding: 20px 24px 32px; }}
  .summary-cards {{
    display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr));
    gap: 14px; margin: 16px 0 18px;
  }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }}
  .card-title {{ font-size: 12px; color: var(--muted); letter-spacing: .02em; text-transform: uppercase; }}
  .card-value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  table {{
    border-collapse: separate; border-spacing: 0; width: 100%; font-size: 14px; background: var(--card);
    border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
  }}
  th, td {{ padding: 10px 12px; vertical-align: top; border-bottom: 1px solid var(--border); }}
  th {{
    text-align: left; background: #f3f4f7; font-weight: 600; position: sticky; top: 0; z-index: 1;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #fbfcff; }}
  td.label {{ max-width: 520px; }}
  td.nums {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; white-space: nowrap; }}
  td.sim {{ text-align: center; font-variant-numeric: tabular-nums; border-left: 1px dashed #eaeaea; }}
  td.sim.strong {{ font-weight: 700; }}
  .legend {{ margin-top: 12px; color: var(--muted); font-size: 12px; }}
  .badge {{
    display:inline-block; margin-left:4px; color:#999; font-weight:700;
    font-size:12px; line-height:1; vertical-align: baseline;
  }}
</style>
</head>
<body>
  <div class="topbar">{html_lib.escape(title)}<div class="subtitle">{html_lib.escape(subtitle)}</div></div>
  <div class="container">
    {summary_html}
    <table>
      {html_header}
      {''.join(html_rows)}
    </table>
    <div class="legend">
      Number sim values with a <span class="badge">•</span> are <em>display fills</em> shown for readability when numeric data is missing; they are not used in the averages above.
      Similarity cells are color-scaled (red → green).
    </div>
  </div>
</body>
</html>"""
    out_html.write_text(html_doc, encoding="utf-8")
    return avg_label, avg_num, avg_over


# ---------- pairing & summary ----------
def _norm_stem(p: Path) -> str:
    s = p.stem
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[\s\-_]+", " ", s).strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _token_set(s: str) -> set[str]:
    return {t for t in re.split(r"\W+", _norm_stem(Path(s))) if t}


def _token_jaccard_name(a: Path, b: Path) -> float:
    A, B = _token_set(a.stem), _token_set(b.stem)
    if not A and not B:
        return 1.0
    return len(A & B) / (len(A | B) or 1)


def _gather_files(root: Path, recursive: bool) -> List[Path]:
    pat = "**/*.md" if recursive else "*.md"
    return sorted(root.glob(pat))


def _safe_slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if ch.isalnum() or ch in (" ", "-", "_", "."))
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "report"


def compute_quick_stats(df: pd.DataFrame) -> Dict[str, Any]:
    gt_present = df["gt_label"].astype(str).str.len() > 0
    res_present = df["res_label"].astype(str).str.len() > 0
    paired = gt_present & res_present
    gt_only = gt_present & ~res_present
    res_only = res_present & ~gt_present
    return {
        "rows_total": int(len(df)),
        "paired_rows": int(paired.sum()),
        "gt_only_rows": int(gt_only.sum()),
        "res_only_rows": int(res_only.sum()),
        "avg_label": (
            float(df["label_sim"].dropna().mean())
            if df["label_sim"].notna().any()
            else np.nan
        ),
        "avg_num": (
            float(df["num_sim"].dropna().mean())
            if df["num_sim"].notna().any()
            else np.nan
        ),
        "avg_over": (
            float(df["overall_sim"].dropna().mean())
            if df["overall_sim"].notna().any()
            else np.nan
        ),
    }


def pair_files(
    gt_files: List[Path], res_files: List[Path]
) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    res_unused = set(res_files)

    # map by normalized stem
    gt_map = {_norm_stem(p): p for p in gt_files}
    res_map = {_norm_stem(p): p for p in res_files}

    # exact normalized stem matches
    for key, g in list(gt_map.items()):
        if key in res_map and res_map[key] in res_unused:
            pairs.append((g, res_map[key], "exact-stem"))
            res_unused.remove(res_map[key])
            del gt_map[key]
            del res_map[key]

    remaining_gt = list(gt_map.values())
    remaining_res = list(res_unused)

    scored: List[Tuple[float, Path, Path]] = []
    for g in remaining_gt:
        for r in remaining_res:
            ratio = difflib.SequenceMatcher(None, _norm_stem(g), _norm_stem(r)).ratio()
            tj = _token_jaccard_name(g, r)
            score = 0.6 * ratio + 0.4 * tj
            scored.append((score, g, r))
    scored.sort(reverse=True, key=lambda x: x[0])

    used_g, used_r = set(), set()
    for score, g, r in scored:
        if g in used_g or r in used_r:
            continue
        if (score >= FUZZY_MIN_RATIO) or (
            _token_jaccard_name(g, r) >= FUZZY_TOKEN_MIN_JACC
        ):
            pairs.append((g, r, f"fuzzy:{score:.3f}"))
            used_g.add(g)
            used_r.add(r)
            if r in res_unused:
                res_unused.remove(r)

    return pairs


def build_index_html(rows: List[Dict[str, Any]], out_path: Path):
    def esc(s: Any) -> str:
        return html_lib.escape("" if s is None else str(s))

    def fmt(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "—"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    tbl_rows = []
    for r in rows:
        link = f"<a href='{esc(r['report_rel'])}'>{esc(Path(r['report_rel']).name)}</a>"
        tbl_rows.append(
            "<tr>"
            f"<td>{esc(r['gt_name'])}</td>"
            f"<td>{esc(r['res_name'])}</td>"
            f"<td>{link}</td>"
            f"<td style='text-align:center'>{fmt(r['avg_over'])}</td>"
            f"<td style='text-align:center'>{fmt(r['avg_label'])}</td>"
            f"<td style='text-align:center'>{fmt(r['avg_num'])}</td>"
            f"<td style='text-align:right'>{fmt(r['rows_total'])}</td>"
            f"<td style='text-align:right'>{fmt(r['paired_rows'])}</td>"
            f"<td style='text-align:right'>{fmt(r['gt_only_rows'])}</td>"
            f"<td style='text-align:right'>{fmt(r['res_only_rows'])}</td>"
            f"<td>{esc(r['pair_reason'])}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>OLMOCR vs Gemin — Index</title>
<style>
  body {{ font-family: Segoe UI, Roboto, Helvetica, Arial, sans-serif; background:#f7f8fb; color:#111; margin:0; }}
  .topbar {{ background:#0b1020; color:#fff; padding:18px 24px; font-size:18px; font-weight:600; }}
  .container {{ padding:20px 24px 32px; }}
  table {{ width:100%; border-collapse:separate; border-spacing:0; background:#fff; border:1px solid #e6e6e6; border-radius:14px; overflow:hidden; }}
  th, td {{ padding:10px 12px; border-bottom:1px solid #e6e6e6; }}
  th {{ background:#f3f4f7; text-align:left; position:sticky; top:0; z-index:1; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:#fbfcff; }}
</style>
</head>
<body>
  <div class="topbar">Markdown Comparison — Index (Gemini folder ↔ OLMOCR folder)</div>
  <div class="container">
    <table>
      <tr>
        <th>Gemini file</th>
        <th>OLMOCR file</th>
        <th>Report</th>
        <th>Avg overall</th>
        <th>Avg label</th>
        <th>Avg numeric</th>
        <th>Total rows</th>
        <th>Paired</th>
        <th>Gemini-only</th>
        <th>OLMOCR-only</th>
        <th>Matched by</th>
      </tr>
      {''.join(tbl_rows)}
    </table>
  </div>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ---------- main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_files = _gather_files(GT_DIR, RECURSIVE_SEARCH)
    res_files = _gather_files(RES_DIR, RECURSIVE_SEARCH)

    if not gt_files:
        print(f"[WARN] No .md files found in GT_DIR: {GT_DIR}")
    if not res_files:
        print(f"[WARN] No .md files found in RES_DIR: {RES_DIR}")

    pairs = pair_files(gt_files, res_files)
    if not pairs:
        print("[WARN] No file pairs found (check names or thresholds).")
    index_rows = []

    for idx, (gtp, resp, reason) in enumerate(pairs, start=1):
        df = build_dataframe(gtp, resp)
        # Per-pair out folder
        slug = _safe_slug(gtp.stem)[:80]
        uniq = hashlib.md5(
            (gtp.name + "||" + resp.name).encode("utf-8", "ignore")
        ).hexdigest()[:8]
        out_subdir = OUT_DIR / f"{idx:03d}_{slug}_{uniq}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        csv_path = out_subdir / "md_compare_unified.csv"
        html_path = out_subdir / "md_compare_report.html"

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        avg_label, avg_num, avg_over = write_html(df, html_path, gtp, resp)

        stats = compute_quick_stats(df)
        index_rows.append(
            {
                "gt_name": gtp.name,
                "res_name": resp.name,
                "report_rel": html_path.relative_to(OUT_DIR).as_posix(),
                "avg_over": stats["avg_over"],
                "avg_label": stats["avg_label"],
                "avg_num": stats["avg_num"],
                "rows_total": stats["rows_total"],
                "paired_rows": stats["paired_rows"],
                "gt_only_rows": stats["gt_only_rows"],
                "res_only_rows": stats["res_only_rows"],
                "pair_reason": reason,
            }
        )

        print(f"[OK] {gtp.name}  ↔  {resp.name}")
        print(f"     -> {csv_path}")
        print(f"     -> {html_path}")
        print(
            f"     Avg: overall={avg_over:.4f} | label={avg_label:.4f} | num={avg_num:.4f}"
        )

    # Write index
    if index_rows:
        idx_csv = OUT_DIR / "index.csv"
        pd.DataFrame(index_rows).to_csv(idx_csv, index=False, encoding="utf-8-sig")
        idx_html = OUT_DIR / "index.html"
        build_index_html(index_rows, idx_html)
        print(f"\nSummary:")
        print(f"  CSV : {idx_csv}")
        print(f"  HTML: {idx_html}")
        try:
            webbrowser.open(idx_html.as_uri())
        except Exception:
            pass
    else:
        print("[INFO] Nothing to summarize.")


if __name__ == "__main__":
    main()
