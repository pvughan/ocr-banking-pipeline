"""
pipeline_engine.py
==================
Xử lý PDF nhiều trang + Dual-engine OCR (Tesseract + Qwen2-VL local).

Luồng (đề bài mục 5.1):
    [1] render_document      PDF -> ảnh >= 200 DPI (pdf2image/poppler) hoặc nạp ảnh JPG/PNG/TIFF
    [2] prepare_page         OpenCV: xoay trang ngược (OSD), cắt viền đen, khử nghiêng, khử bóng mép
                             gáy, triệt tem đỏ (kênh R), khử nhiễu, CLAHE
    [3] run_tesseract        Tesseract vie: TSV (bbox + conf từng từ) + full text của từng trang
    [4] TesseractFieldExtractor  dò `anchor` của template trong TSV, trích giá trị + confidence
    [5] QwenVLEngine         Qwen2-VL-2B-Instruct local, prompt JSON-schema sinh từ template,
                             kèm gợi ý OCR của Tesseract; CPU/float32, ảnh thu nhỏ để suy luận nhanh
    [6] OCRPipeline.run      điều phối toàn bộ -> ExtractionRun (đầu vào của fusion.py)

Ghi chú quan trọng về Qwen trên CPU
-----------------------------------
* Ảnh trang đưa vào Qwen bị giới hạn 224*224 <= pixels <= 448*448 (ràng buộc phần cứng) nên chữ
  nhỏ KHÔNG đọc được trực tiếp từ ảnh – Qwen chủ yếu hiểu BỐ CỤC/NGỮ NGHĨA; vì vậy prompt kèm gợi ý
  OCR quanh anchor, và có bước `read_region` (cắt vùng quanh anchor -> đọc lại với độ phân giải
  hiệu dụng cao hơn nhiều) cho các trường Qwen bỏ sót.
* Toàn bộ chạy OFFLINE: HF_HUB_OFFLINE=1, local_files_only=True. Không có lời gọi mạng nào.
"""
from __future__ import annotations

import io
import json
import logging
import os
import platform
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageSequence
from pytesseract import Output
from rapidfuzz import fuzz

from config import PIPELINE_VERSION, PROMPT_VERSION, PROMPTS, AppConfig, PreprocessConfig, QwenConfig, \
    RenderConfig, TesseractConfig
from template_parser import (FieldSpec, Template, clean_whitespace, fold_vietnamese, repair_digits,
                             validator_search_regex)

log = logging.getLogger("ocr.pipeline")
ProgressFn = Callable[[str, float, str], None]


class PipelineError(RuntimeError):
    """Lỗi nghiệp vụ của pipeline (thiếu poppler/tesseract, PDF hỏng, không tải được model...)."""


# --------------------------------------------------------------------------- #
# Mô hình dữ liệu kết quả
# --------------------------------------------------------------------------- #
@dataclass
class PageImage:
    page_no: int            # 1-based
    image: Image.Image      # RGB
    dpi: int


@dataclass
class PreparedPage:
    page_no: int
    gray: np.ndarray        # ảnh xám đã tiền xử lý -> đưa vào Tesseract
    color: Image.Image      # ảnh màu đã chỉnh hình học (cùng toạ độ với gray) -> đưa vào Qwen
    notes: List[str] = field(default_factory=list)
    rotation: int = 0
    skew_deg: float = 0.0
    scale: float = 1.0


@dataclass
class TessWord:
    text: str
    conf: float             # 0-100
    left: int
    top: int
    width: int
    height: int
    block: int
    par: int
    line: int
    word_num: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


@dataclass
class TessLine:
    words: List[TessWord]
    block: int
    par: int

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def conf(self) -> float:
        """Confidence trung bình 0-1."""
        return float(np.mean([w.conf for w in self.words])) / 100.0 if self.words else 0.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return _union_bbox([w.bbox for w in self.words])


@dataclass
class PageOCR:
    page_no: int
    text: str
    lines: List[TessLine]
    words: List[TessWord]
    mean_conf: float        # 0-1
    psm: int
    lang: str
    size: Tuple[int, int]   # (w, h) của ảnh đưa vào Tesseract
    latency_s: float
    notes: List[str] = field(default_factory=list)


@dataclass
class TesseractReading:
    value: Optional[str]                    # None cho trường table
    confidence: float                       # trung bình conf các từ cấu thành (0-1)
    page: int
    bbox: Optional[Tuple[int, int, int, int]]
    anchor_score: float                     # 0-1 (0 = không dùng anchor)
    context: str = ""                       # đoạn văn bản quanh anchor (làm gợi ý cho Qwen)
    evidence: str = ""
    rows: Optional[List[List[str]]] = None  # trường table: các hàng -> các ô


@dataclass
class QwenReading:
    value: Any                              # str | list[dict] | None
    page: int
    source: str = "page"                    # page | region
    raw_json: str = ""


@dataclass
class FieldReadings:
    tesseract: Optional[TesseractReading] = None
    qwen: Optional[QwenReading] = None


@dataclass
class ExtractionRun:
    document_name: str
    template_name: str
    n_pages: int
    pages: List[PageOCR]
    readings: Dict[str, FieldReadings]
    qwen_outputs: List[Dict[str, Any]]
    timings: Dict[str, Any]
    environment: Dict[str, Any]
    config: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    page_images: List[Image.Image] = field(default_factory=list, repr=False)


def _union_bbox(boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


# --------------------------------------------------------------------------- #
# Phát hiện công cụ ngoài (macOS Homebrew)
# --------------------------------------------------------------------------- #
_BREW_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")


def _find_executable(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for d in _BREW_DIRS:
        cand = Path(d) / name
        if cand.exists():
            return str(cand)
    return None


def configure_tesseract() -> str:
    """Trỏ pytesseract tới binary Tesseract (Homebrew Apple Silicon: /opt/homebrew/bin/tesseract)."""
    exe = _find_executable("tesseract")
    if not exe:
        raise PipelineError("Không tìm thấy Tesseract. Cài bằng: brew install tesseract tesseract-lang")
    pytesseract.pytesseract.tesseract_cmd = exe
    return exe


def tesseract_version() -> str:
    try:
        configure_tesseract()
        return str(pytesseract.get_tesseract_version())
    except Exception as exc:  # pragma: no cover
        return f"không xác định ({exc})"


def resolve_tesseract_lang(cfg: TesseractConfig) -> Tuple[str, List[str]]:
    """Chọn ngôn ngữ khả dụng. Thiếu 'vie' -> fallback (kèm cảnh báo rõ ràng, KHÔNG im lặng)."""
    configure_tesseract()
    warnings: List[str] = []
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        available = set()
    wanted = cfg.lang.split("+")
    if available and not all(w in available for w in wanted):
        if cfg.fallback_lang in available:
            warnings.append(f"Tesseract chưa có gói ngôn ngữ '{cfg.lang}' -> dùng '{cfg.fallback_lang}'. "
                            f"Chất lượng tiếng Việt sẽ KÉM; cài: brew install tesseract-lang")
            return cfg.fallback_lang, warnings
        raise PipelineError(f"Tesseract không có ngôn ngữ '{cfg.lang}' (có: {sorted(available)}).")
    return cfg.lang, warnings


# --------------------------------------------------------------------------- #
# [1] RENDER PDF / NẠP ẢNH
# --------------------------------------------------------------------------- #
def _poppler_dir() -> Optional[str]:
    exe = _find_executable("pdftoppm")
    return str(Path(exe).parent) if exe else None


def render_document(data: bytes, filename: str, cfg: RenderConfig) -> List[PageImage]:
    """PDF (2..N trang) hoặc ảnh -> danh sách PageImage RGB. PDF render với dpi >= 200."""
    name = (filename or "").lower()
    is_pdf = data[:5] == b"%PDF-" or name.endswith(".pdf")
    if is_pdf:
        try:
            from pdf2image import convert_from_bytes, pdfinfo_from_bytes
            from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
        except ImportError as exc:  # pragma: no cover
            raise PipelineError("Thiếu thư viện pdf2image: pip install pdf2image") from exc
        poppler = cfg.poppler_path or _poppler_dir()
        try:
            info = pdfinfo_from_bytes(data, poppler_path=poppler)
            n_pages = int(info.get("Pages", 0))
            if n_pages <= 0:
                raise PipelineError("PDF không có trang nào.")
            if n_pages > cfg.max_pages:
                raise PipelineError(f"PDF có {n_pages} trang, vượt giới hạn {cfg.max_pages} trang.")
            images = convert_from_bytes(data, dpi=cfg.dpi, thread_count=2, poppler_path=poppler)
        except PDFInfoNotInstalledError as exc:
            raise PipelineError("Chưa cài poppler (pdf2image cần). Cài bằng: brew install poppler") from exc
        except (PDFPageCountError, PDFSyntaxError) as exc:
            raise PipelineError(f"File PDF hỏng hoặc không đọc được: {exc}") from exc
        return [PageImage(i + 1, im.convert("RGB"), cfg.dpi) for i, im in enumerate(images)]

    try:
        img = Image.open(io.BytesIO(data))
        frames = [f.copy() for f in ImageSequence.Iterator(img)][: cfg.max_pages]
    except Exception as exc:
        raise PipelineError(f"Không đọc được file ảnh: {exc}") from exc
    pages: List[PageImage] = []
    for i, fr in enumerate(frames):
        fr = ImageOps.exif_transpose(fr).convert("RGB")
        dpi_info = img.info.get("dpi")
        dpi = int(dpi_info[0]) if dpi_info and dpi_info[0] and dpi_info[0] > 30 else \
            int(round(fr.width / 8.27))  # ước lượng theo khổ A4 ngang 8.27 inch
        pages.append(PageImage(i + 1, fr, max(dpi, 50)))
    return pages


# --------------------------------------------------------------------------- #
# [2] TIỀN XỬ LÝ ẢNH (OpenCV)
# --------------------------------------------------------------------------- #
def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _rotate_small(img: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    """Xoay quanh tâm (góc dương = ngược chiều kim đồng hồ), giữ nguyên kích thước.
    `border`: giá trị lấp viền (255 = nền trắng; dùng 0 cho ảnh nhị phân đã đảo bit, mực = 255)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    value = (border, border, border) if img.ndim == 3 else border
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=value)


def _confident_word_count(gray: np.ndarray, lang: str) -> int:
    """Số từ OCR có conf >= 60 trên ảnh thu nhỏ – thước đo nhanh 'ảnh có đúng chiều không'."""
    d = pytesseract.image_to_data(gray, lang=lang, config="--psm 6", output_type=Output.DICT, timeout=40)
    return sum(1 for t, c in zip(d["text"], d["conf"]) if (t or "").strip() and float(c) >= 60)


def detect_page_rotation(bgr: np.ndarray, lang: str = "eng") -> int:
    """
    Phát hiện trang scan ngược (0/90/180/270, chiều kim đồng hồ) bằng Tesseract OSD.
    Trang thưa chữ làm OSD kém tin cậy -> kiểm chứng bằng OCR nhanh cả hai hướng.
    """
    try:
        h, w = bgr.shape[:2]
        scale = 1400.0 / max(h, w)
        small = cv2.cvtColor(cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr,
                             cv2.COLOR_BGR2GRAY)
        osd = pytesseract.image_to_osd(small, config="--psm 0", output_type=Output.DICT, timeout=30)
        rotate, conf = int(osd.get("rotate", 0)), float(osd.get("orientation_conf", 0))
        if rotate not in (90, 180, 270) or conf < 1.0:
            return 0
        if conf >= 5.0:
            return rotate
        code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate]
        n_rot = _confident_word_count(cv2.rotate(small, code), lang)
        n_0 = _confident_word_count(small, lang)
        return rotate if n_rot > 1.2 * max(n_0, 1) else 0
    except Exception as exc:  # thiếu osd.traineddata / trang gần như trống
        log.debug("OSD không khả dụng: %s", exc)
        return 0


def crop_dark_border(bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Cắt viền đen quanh trang scan/chụp (chỉ khi 4 góc thực sự tối)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    p = max(4, int(min(h, w) * 0.03))
    corners = [gray[:p, :p], gray[:p, -p:], gray[-p:, :p], gray[-p:, -p:]]
    if np.mean([c.mean() for c in corners]) > 90:
        return bgr, False
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr, False
    x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if cw * ch < 0.5 * w * h or cw * ch > 0.995 * w * h:
        return bgr, False
    m = 4
    return bgr[y + m:y + ch - m, x + m:x + cw - m], True


def estimate_skew(gray: np.ndarray, max_deg: float = 5.0) -> float:
    """Ước lượng độ nghiêng (độ) bằng projection profile: góc làm phương sai tổng theo hàng lớn nhất."""
    h, w = gray.shape[:2]
    scale = 900.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)

    def score(angle: float) -> float:
        rot = _rotate_small(binary, angle, border=0)
        return float(np.var(rot.sum(axis=1) / 255.0))

    coarse = np.arange(-max_deg, max_deg + 1e-6, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.8, best + 0.8 + 1e-6, 0.2)
    best = max(fine, key=score)
    return float(best) if abs(best) >= 0.2 else 0.0


def prepare_page(page: PageImage, cfg: PreprocessConfig, lang: str = "eng") -> PreparedPage:
    """Tiền xử lý một trang. Mọi phép chỉnh hình học áp dụng cho CẢ ảnh màu (Qwen) và ảnh xám (Tesseract)."""
    bgr = _pil_to_bgr(page.image)
    notes: List[str] = []
    rotation, skew = 0, 0.0
    scale = 1.0

    if not cfg.enabled:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return PreparedPage(page.page_no, gray, page.image, ["tắt tiền xử lý"])

    if cfg.auto_orient:
        rotation = detect_page_rotation(bgr, lang)
        if rotation:
            code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation]
            bgr = cv2.rotate(bgr, code)
            notes.append(f"xoay trang {rotation}° (OSD)")

    if cfg.crop_border:
        bgr, cropped = crop_dark_border(bgr)
        if cropped:
            notes.append("cắt viền đen")

    if page.dpi < cfg.target_dpi:
        scale = min(cfg.target_dpi / max(page.dpi, 1), cfg.max_upscale)
        if scale > 1.05:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            notes.append(f"upscale x{scale:.2f} (DPI {page.dpi} -> {int(page.dpi * scale)})")
        else:
            scale = 1.0

    if cfg.deskew:
        skew = estimate_skew(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cfg.max_deskew_deg)
        if skew:
            bgr = _rotate_small(bgr, skew)
            notes.append(f"khử nghiêng {skew:+.1f}°")

    # Ảnh xám cho Tesseract: kênh R làm tem/dấu đỏ chuyển thành vùng sáng (gần như biến mất).
    gray = bgr[:, :, 2].copy() if cfg.suppress_red else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if cfg.suppress_red:
        notes.append("triệt tem đỏ (kênh R)")

    if cfg.normalize_background:   # khử bóng mép gáy / chụp điện thoại
        k = max(15, int(gray.shape[1] / 60)) | 1
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
        gray = cv2.divide(gray, bg, scale=255)
        notes.append("chuẩn hoá nền")
    if cfg.denoise:
        gray = cv2.medianBlur(gray, 3)
    if cfg.clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if cfg.adaptive_threshold:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        notes.append("nhị phân hoá thích nghi")

    return PreparedPage(page.page_no, gray, _bgr_to_pil(bgr), notes, rotation, skew, scale)


# --------------------------------------------------------------------------- #
# [3] TESSERACT
# --------------------------------------------------------------------------- #
def _build_page_ocr(data: Dict[str, List[Any]], page_no: int, psm: int, lang: str,
                    size: Tuple[int, int], latency: float) -> PageOCR:
    words: List[TessWord] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 0:
            continue
        words.append(TessWord(text, conf, int(data["left"][i]), int(data["top"][i]), int(data["width"][i]),
                              int(data["height"][i]), int(data["block_num"][i]), int(data["par_num"][i]),
                              int(data["line_num"][i]), int(data["word_num"][i])))
    lines: List[TessLine] = []
    current_key = None
    for w in words:
        key = (w.block, w.par, w.line)
        if key != current_key:
            lines.append(TessLine([], w.block, w.par))
            current_key = key
        lines[-1].words.append(w)
    parts: List[str] = []
    prev_par = None
    for ln in lines:
        par = (ln.block, ln.par)
        if prev_par is not None and par != prev_par:
            parts.append("")
        parts.append(ln.text)
        prev_par = par
    mean_conf = float(np.mean([w.conf for w in words])) / 100.0 if words else 0.0
    return PageOCR(page_no, "\n".join(parts), lines, words, mean_conf, psm, lang, size, latency)


def run_tesseract(prepared: PreparedPage, cfg: TesseractConfig, lang: str) -> PageOCR:
    """
    Chạy Tesseract một lần/PSM và lấy TSV (bbox + conf từng từ). Full text được dựng lại từ chính TSV
    (cùng thứ tự đọc của Tesseract) nên không phải chạy OCR hai lần -> tiết kiệm ~50% thời gian CPU.
    Nếu cấu hình nhiều PSM, chọn PSM có confidence trung bình cao nhất.
    """
    configure_tesseract()
    img = Image.fromarray(prepared.gray)
    best: Optional[PageOCR] = None
    for psm in cfg.psm_list:
        t0 = time.perf_counter()
        try:
            data = pytesseract.image_to_data(
                img, lang=lang, output_type=Output.DICT, timeout=cfg.timeout_s,
                config=f"--oem {cfg.oem} --psm {psm} -c preserve_interword_spaces=1")
        except RuntimeError as exc:           # timeout
            raise PipelineError(f"Tesseract quá thời gian ở trang {prepared.page_no} (PSM {psm}): {exc}") from exc
        except pytesseract.TesseractError as exc:
            raise PipelineError(f"Tesseract lỗi ở trang {prepared.page_no}: {exc}") from exc
        res = _build_page_ocr(data, prepared.page_no, psm, lang, img.size, time.perf_counter() - t0)
        if best is None or res.mean_conf > best.mean_conf:
            best = res
    assert best is not None
    best.notes = list(prepared.notes)
    return best


# --------------------------------------------------------------------------- #
# [4] TRÍCH TRƯỜNG TỪ TSV THEO ANCHOR
# --------------------------------------------------------------------------- #
_LEADING_JUNK = re.compile(r"^[\s:;.,\-_–—|/\\]+")


class TesseractFieldExtractor:
    """
    Với mỗi trường: dò `anchor` (so khớp mờ trên dạng bỏ dấu để chịu lỗi OCR) trong các dòng TSV, rồi
    lấy giá trị phía sau anchor trên cùng dòng / các dòng kế tiếp. Trường có regex/validator (CCCD,
    MST, ngày, tiền, email...) dùng *regex-guided search* trong cửa sổ quanh anchor – cách rẻ và
    chính xác nhất để lấy các trường mã số. Confidence = trung bình conf từng từ cấu thành.
    """

    def __init__(self, pages: List[PageOCR], cfg: TesseractConfig):
        self.pages = pages
        self.cfg = cfg

    # ---- dò anchor ------------------------------------------------------
    def _anchor_hits(self, page: PageOCR, anchor: str) -> List[Tuple[int, List[TessWord], int, float, int]]:
        """Trả về (line_idx, words_seq, end_idx_in_seq, score 0-100, first_line_after)."""
        a_fold = fold_vietnamese(anchor)
        if not a_fold:
            return []
        k = len(a_fold.split())
        hits = []
        for li, line in enumerate(page.lines):
            nxt = page.lines[li + 1] if li + 1 < len(page.lines) else None
            variants = [(line.words, 0)]
            if nxt is not None and k >= 3:               # anchor dài có thể bị ngắt xuống dòng
                variants.append((line.words + nxt.words, len(line.words)))
            for seq, split in variants:
                folded = [(i, fold_vietnamese(w.text)) for i, w in enumerate(seq)]
                folded = [(i, f) for i, f in folded if f]
                best_score, best_end = 0.0, -1
                for s in range(0, max(1, len(folded) - k + 1)):
                    win = folded[s:s + k]
                    if len(win) < k:
                        break
                    score = fuzz.ratio(a_fold, " ".join(f for _, f in win))
                    if score > best_score:
                        best_score, best_end = score, win[-1][0]
                if best_score >= self.cfg.anchor_min_score:
                    after = li + 2 if (split and best_end >= split) else li + 1
                    hits.append((li, seq, best_end, best_score, after))
                    break
        return hits

    # ---- tiện ích cửa sổ văn bản ---------------------------------------
    @staticmethod
    def _join_with_spans(words: List[TessWord]) -> Tuple[str, List[Tuple[int, int, TessWord]]]:
        """
        Ghép từ thành chuỗi + bản đồ vị trí ký tự -> từ. Các từ toàn chữ số liền nhau ('0909 123 456')
        được nối KHÔNG khoảng trắng để regex mã số/điện thoại khớp được dù Tesseract tách nhóm số.
        """
        text, spans = "", []
        prev_numeric = False
        for w in words:
            numeric = re.fullmatch(r"[\d.,\-]+", w.text) is not None
            if text and not (numeric and prev_numeric):
                text += " "
            spans.append((len(text), len(text) + len(w.text), w))
            text += w.text
            prev_numeric = numeric
        return text, spans

    def _build_reading_from_words(self, words: List[TessWord], page: int, anchor_score: float,
                                  context: str, evidence: str) -> Optional[TesseractReading]:
        words = [w for w in words if w.text.strip()]
        if not words:
            return None
        value = clean_whitespace(" ".join(w.text for w in words))
        value = _LEADING_JUNK.sub("", value).strip()
        if not value:
            return None
        return TesseractReading(value, float(np.mean([w.conf for w in words])) / 100.0, page,
                                _union_bbox([w.bbox for w in words]), anchor_score, context, evidence)

    # ---- trích 1 trường -------------------------------------------------
    def extract(self, spec: FieldSpec) -> Optional[TesseractReading]:
        if spec.is_table:
            return self.extract_table(spec)
        n_pages = len(self.pages)
        hint_pages = set(spec.page_candidates(n_pages))
        pattern = validator_search_regex(spec.validation, spec.type)
        candidates: List[Tuple[float, TesseractReading]] = []
        for page in self.pages:
            if spec.anchor:
                for li, seq, end, score, after in self._anchor_hits(page, spec.anchor):
                    rd = self._read_after_anchor(spec, page, seq, end, after, score, pattern)
                    if rd:
                        s = 0.45 * rd.anchor_score + 0.35 * rd.confidence + 0.10 + \
                            (0.10 if page.page_no in hint_pages and len(hint_pages) < n_pages else 0.0)
                        candidates.append((s, rd))
            elif pattern is not None:
                rd = self._search_whole_page(spec, page, pattern)
                if rd:
                    candidates.append((0.35 * rd.confidence + 0.10 +
                                       (0.10 if page.page_no in hint_pages else 0.0), rd))
        if not candidates:
            return None
        return max(candidates, key=lambda t: t[0])[1]

    def _read_after_anchor(self, spec: FieldSpec, page: PageOCR, seq: List[TessWord], end: int, after: int,
                           score: float, pattern: Optional[re.Pattern]) -> Optional[TesseractReading]:
        remainder = seq[end + 1:]
        nxt_lines = page.lines[after: after + self.cfg.lookahead_lines + (1 if pattern else 0)]
        next_words = [w for ln in nxt_lines for w in ln.words]
        anchor_line_words = seq
        ctx_words = remainder + next_words
        context = clean_whitespace(" ".join(w.text for w in ctx_words))[:300]
        a_score = score / 100.0
        ev = f"anchor '{spec.anchor}' khớp {score:.0f}% ở trang {page.page_no}"

        if spec.type in ("checkbox", "boolean"):
            words = anchor_line_words + next_words[:8]
            return self._build_reading_from_words(words, page.page_no, a_score, context, ev + "; vùng checkbox")

        if pattern is not None:
            window = remainder + next_words
            text, spans = self._join_with_spans(window)
            if spec.type in ("number", "money", "date") or "msdn" in spec.validation.lower() \
                    or "cccd" in spec.validation.lower():
                text = repair_digits(text)       # ánh xạ 1-1 ký tự nên spans vẫn đúng
            m = pattern.search(text)
            if not m:
                return None
            used = [w for (a, b, w) in spans if a < m.end() and b > m.start()]
            rd = self._build_reading_from_words(used, page.page_no, a_score, context, ev + "; regex-guided")
            if rd:
                rd.value = clean_whitespace(m.group(0))
            return rd

        # văn bản tự do: phần còn lại của dòng; nếu trống lấy dòng kế; địa chỉ có thể kéo dài 2 dòng
        words = list(remainder)
        text_len = len(clean_whitespace(" ".join(w.text for w in words)))
        multiline = "dia_chi" in spec.field_key or "địa chỉ" in spec.field_name.lower()
        if text_len < 2 and nxt_lines:
            words = list(nxt_lines[0].words)
            if multiline and len(nxt_lines) > 1 and ":" not in nxt_lines[1].text:
                words += nxt_lines[1].words
        elif multiline and nxt_lines and ":" not in nxt_lines[0].text and \
                clean_whitespace(" ".join(w.text for w in words)).endswith(","):
            words += nxt_lines[0].words
        return self._build_reading_from_words(words, page.page_no, a_score, context, ev)

    def _search_whole_page(self, spec: FieldSpec, page: PageOCR, pattern: re.Pattern) -> Optional[TesseractReading]:
        text, spans = self._join_with_spans(page.words)
        if spec.type in ("number", "money", "date"):
            text = repair_digits(text)
        m = pattern.search(text)
        if not m:
            return None
        used = [w for (a, b, w) in spans if a < m.end() and b > m.start()]
        rd = self._build_reading_from_words(used, page.page_no, 0.0, "", "regex-guided toàn trang (không anchor)")
        if rd:
            rd.value = clean_whitespace(m.group(0))
        return rd

    # ---- bảng -----------------------------------------------------------
    def extract_table(self, spec: FieldSpec) -> Optional[TesseractReading]:
        """
        Dựng hàng/ô từ TSV: lấy các dòng bên dưới anchor tới khi gặp khoảng trống dọc lớn, tách ô theo
        khoảng cách ngang giữa các từ. Chỉ dùng để ĐỐI CHIẾU cấu trúc (số hàng/cột) với Qwen.
        """
        best: Optional[TesseractReading] = None
        for page in self.pages:
            hits = self._anchor_hits(page, spec.anchor) if spec.anchor else []
            for li, seq, end, score, after in hits[:2]:
                lines = page.lines[after:after + self.cfg.table_max_rows]
                if not lines:
                    continue
                heights = [w.height for ln in lines for w in ln.words] or [20]
                med_h = float(np.median(heights))
                rows: List[List[str]] = []
                prev_bottom: Optional[int] = None
                used_words: List[TessWord] = []
                for ln in lines:
                    x0, y0, x1, y1 = ln.bbox
                    if prev_bottom is not None and y0 - prev_bottom > 3.0 * med_h:
                        break
                    prev_bottom = y1
                    cells, cur, last_right = [], [], None
                    for w in ln.words:
                        if last_right is not None and w.left - last_right > 2.2 * med_h:
                            cells.append(" ".join(c.text for c in cur))
                            cur = []
                        cur.append(w)
                        last_right = w.left + w.width
                    if cur:
                        cells.append(" ".join(c.text for c in cur))
                    if len(cells) >= 2:
                        rows.append(cells)
                        used_words += ln.words
                if spec.columns and rows:        # bỏ hàng tiêu đề nếu khớp tên cột
                    head = fold_vietnamese(" ".join(rows[0]))
                    cols = fold_vietnamese(" ".join(c.column_name or c.column_key for c in spec.columns))
                    if fuzz.token_set_ratio(head, cols) >= 70:
                        rows = rows[1:]
                if rows:
                    conf = float(np.mean([w.conf for w in used_words])) / 100.0
                    rd = TesseractReading(None, conf, page.page_no, _union_bbox([w.bbox for w in used_words]),
                                          score / 100.0, "", f"bảng dựng từ TSV: {len(rows)} hàng", rows=rows)
                    if best is None or len(rows) > len(best.rows or []):
                        best = rd
        return best


# --------------------------------------------------------------------------- #
# [5] QWEN2-VL LOCAL
# --------------------------------------------------------------------------- #
def _type_hint(spec: FieldSpec) -> str:
    t = spec.type
    if spec.enum_values:
        return "chọn đúng một giá trị trong: " + " | ".join(spec.enum_values)
    if t == "date":
        return "ngày, chép nguyên như trên tài liệu"
    if t == "money":
        return "số tiền, chép nguyên như trên tài liệu"
    if t == "number":
        return "số"
    if t in ("checkbox", "boolean"):
        return "true nếu được đánh dấu, false nếu không"
    if t == "table":
        return "bảng; cột: " + ", ".join(c.column_key for c in spec.columns)
    return "văn bản"


def build_page_prompt(specs: Sequence[FieldSpec], hints: Dict[str, str], page_text: str, cfg: QwenConfig) -> str:
    """Sinh prompt (đã VERSION trong config.PROMPTS) + JSON schema từ danh sách trường của template."""
    guide_lines, schema = [], {}
    for s in specs:
        line = f'- "{s.key}": {s.field_name} ({_type_hint(s)})'
        if s.anchor:
            line += f'; nhãn gần vị trí: "{s.anchor}"'
        hint = hints.get(s.key, "")
        if hint:
            line += f'; OCR gần nhãn: "{hint[:cfg.hint_chars]}"'
        guide_lines.append(line)
        schema[s.key] = [{c.column_key: None for c in s.columns} or {"...": None}] if s.is_table else None
    block = ""
    if page_text and cfg.page_text_chars > 0:
        block = PROMPTS["page_text_block"].format(page_text=page_text[:cfg.page_text_chars])
    return PROMPTS["page_instruction"].format(field_guide="\n".join(guide_lines),
                                              schema=json.dumps(schema, ensure_ascii=False),
                                              page_text_block=block)


def parse_json_loose(text: str, keys: Sequence[str]) -> Dict[str, Any]:
    """Parse JSON từ đầu ra LLM nhỏ: bỏ rào ```json, sửa dấu phẩy thừa/nháy đơn, cứu từng key nếu hỏng."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    candidates = [t[start:end + 1]] if start != -1 and end > start else []
    if start != -1 and end <= start:
        candidates.append(t[start:] + "}")
    for cand in candidates:
        for fixer in (lambda s: s, lambda s: re.sub(r",\s*([}\]])", r"\1", s),
                      lambda s: re.sub(r",\s*([}\]])", r"\1", s.replace("'", '"'))):
            try:
                obj = json.loads(fixer(cand))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    out: Dict[str, Any] = {}
    for k in keys:   # cứu từng key dạng chuỗi/null
        m = re.search(rf'"{re.escape(k)}"\s*:\s*("(?:[^"\\]|\\.)*"|null|true|false)', t)
        if m:
            try:
                out[k] = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return out


class QwenVLEngine:
    """
    Qwen2-VL chạy local bằng transformers: CPU, float32, OFFLINE.
    Model được nạp lười (lần gọi đầu tiên) và dùng lại giữa các tài liệu.
    """

    def __init__(self, cfg: QwenConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None
        self.load_time_s: float = 0.0
        self._torch = None

    # ---- nạp model ------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")          # không bao giờ gọi mạng
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        t0 = time.perf_counter()
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise PipelineError("Thiếu thư viện: pip install torch transformers accelerate") from exc
        self._torch = torch
        threads = self.cfg.num_threads or (os.cpu_count() or 4)
        torch.set_num_threads(threads)
        kwargs = dict(local_files_only=self.cfg.local_files_only)
        try:
            try:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.cfg.model_id, torch_dtype=torch.float32, device_map="cpu", **kwargs)
            except (ValueError, ImportError, TypeError):      # thiếu accelerate -> nạp rồi .to("cpu")
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.cfg.model_id, torch_dtype=torch.float32, **kwargs).to("cpu")
            processor = AutoProcessor.from_pretrained(
                self.cfg.model_id, min_pixels=self.cfg.min_pixels, max_pixels=self.cfg.max_pixels, **kwargs)
        except OSError as exc:
            raise PipelineError(
                f"Không tìm thấy weights '{self.cfg.model_id}' trong Hugging Face cache (chế độ offline). "
                f"Chi tiết: {exc}") from exc
        model.eval()
        self._model, self._processor = model, processor
        self.load_time_s = time.perf_counter() - t0

    def info(self) -> Dict[str, Any]:
        return {"model_id": self.cfg.model_id, "device": "cpu", "dtype": "float32",
                "min_pixels": self.cfg.min_pixels, "max_pixels": self.cfg.max_pixels,
                "thumb_max_px": self.cfg.thumb_max, "prompt_version": PROMPT_VERSION,
                "load_time_s": round(self.load_time_s, 2),
                "torch_threads": self._torch.get_num_threads() if self._torch else None}

    # ---- sinh văn bản ----------------------------------------------------
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        self.load()
        torch, model, processor = self._torch, self._model, self._processor
        img = image.copy().convert("RGB")
        img.thumbnail((self.cfg.thumb_max, self.cfg.thumb_max), Image.LANCZOS)   # ảnh tạm nhỏ -> CPU nhanh
        messages = [{"role": "system", "content": PROMPTS["system"]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     repetition_penalty=1.05)
        trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # ---- API cho pipeline ------------------------------------------------
    def extract_page(self, image: Image.Image, specs: Sequence[FieldSpec], hints: Dict[str, str],
                     page_text: str) -> Tuple[Dict[str, Any], str]:
        prompt = build_page_prompt(specs, hints, page_text, self.cfg)
        n_table = sum(1 for s in specs if s.is_table)
        budget = min(self.cfg.max_new_tokens + 220 * n_table, 24 + 34 * (len(specs) - n_table) + 260 * n_table)
        raw = self.generate(image, prompt, max_new_tokens=max(budget, 64))
        return parse_json_loose(raw, [s.key for s in specs]), raw

    def read_region(self, crop: Image.Image, spec: FieldSpec) -> Tuple[Optional[str], str]:
        prompt = PROMPTS["region_instruction"].format(field_name=spec.field_name, type_hint=_type_hint(spec))
        raw = self.generate(crop, prompt, max_new_tokens=64)
        val = parse_json_loose(raw, ["value"]).get("value")
        return (None if val is None else str(val)), raw


# --------------------------------------------------------------------------- #
# [6] ĐIỀU PHỐI
# --------------------------------------------------------------------------- #
def collect_environment(cfg: AppConfig, qwen: Optional[QwenVLEngine]) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "tesseract": tesseract_version(), "tesseract_lang": cfg.tesseract.lang,
        "mode": "CPU only (float32, không CUDA, không gọi API ngoài)",
        "pipeline_version": PIPELINE_VERSION, "prompt_version": PROMPT_VERSION,
    }
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        env["torch"] = "chưa cài"
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        env["transformers"] = "chưa cài"
    if qwen is not None:
        env["qwen"] = qwen.info()
    return env


class OCRPipeline:
    """Điều phối: render -> tiền xử lý -> Tesseract -> trích theo anchor -> Qwen2-VL -> ExtractionRun."""

    def __init__(self, cfg: Optional[AppConfig] = None, qwen_engine: Optional[Any] = None):
        self.cfg = cfg or AppConfig()
        self.qwen = qwen_engine or QwenVLEngine(self.cfg.qwen)

    @staticmethod
    def _emit(progress: Optional[ProgressFn], stage: str, frac: float, msg: str) -> None:
        if progress:
            progress(stage, max(0.0, min(1.0, frac)), msg)

    def run(self, data: bytes, filename: str, template: Template,
            progress: Optional[ProgressFn] = None, keep_page_images: bool = True) -> ExtractionRun:
        t_total = time.perf_counter()
        warnings: List[str] = []
        timings: Dict[str, Any] = {}

        # [1] Render
        self._emit(progress, "render", 0.0, "Render PDF thành ảnh...")
        t0 = time.perf_counter()
        pages_img = render_document(data, filename, self.cfg.render)
        timings["render_s"] = time.perf_counter() - t0
        n = len(pages_img)
        if n < 1:
            raise PipelineError("Tài liệu không có trang nào.")
        self._emit(progress, "render", 1.0, f"Đã render {n} trang @ {self.cfg.render.dpi} DPI")

        # [2]+[3] Tiền xử lý + Tesseract
        lang, lang_warnings = resolve_tesseract_lang(self.cfg.tesseract)
        warnings += lang_warnings
        prepared: List[PreparedPage] = []
        pages_ocr: List[PageOCR] = []
        t_pre = t_tess = 0.0
        for i, pg in enumerate(pages_img):
            self._emit(progress, "tesseract", i / n, f"Tesseract trang {pg.page_no}/{n} (TSV + confidence)...")
            t0 = time.perf_counter()
            pp = prepare_page(pg, self.cfg.preprocess, lang)
            t_pre += time.perf_counter() - t0
            t0 = time.perf_counter()
            ocr = run_tesseract(pp, self.cfg.tesseract, lang)
            t_tess += time.perf_counter() - t0
            prepared.append(pp)
            pages_ocr.append(ocr)
            if ocr.mean_conf < 0.55:
                warnings.append(f"Trang {pg.page_no}: confidence Tesseract thấp ({ocr.mean_conf:.0%}) – "
                                f"ảnh mờ/nghiêng/tem đè.")
        timings["preprocess_s"], timings["tesseract_s"] = t_pre, t_tess
        timings["tesseract_per_page_s"] = [round(p.latency_s, 2) for p in pages_ocr]
        self._emit(progress, "tesseract", 1.0, "Hoàn tất Tesseract")

        # [4] Trích trường từ TSV
        extractor = TesseractFieldExtractor(pages_ocr, self.cfg.tesseract)
        readings: Dict[str, FieldReadings] = {f.key: FieldReadings() for f in template.fields}
        for spec in template.fields:
            try:
                readings[spec.key].tesseract = extractor.extract(spec)
            except Exception as exc:     # một trường lỗi không làm hỏng cả tài liệu
                log.exception("Tesseract extractor lỗi ở trường %s", spec.key)
                warnings.append(f"Trích Tesseract lỗi ở '{spec.key}': {exc}")

        # [5] Qwen2-VL
        qwen_outputs: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        self._run_qwen(template, readings, prepared, pages_ocr, qwen_outputs, warnings, progress)
        timings["qwen_s"] = time.perf_counter() - t0
        timings["qwen_load_s"] = getattr(self.qwen, "load_time_s", 0.0)
        timings["qwen_calls"] = len(qwen_outputs)
        timings["pipeline_s"] = time.perf_counter() - t_total

        return ExtractionRun(
            document_name=filename, template_name=template.name, n_pages=n, pages=pages_ocr, readings=readings,
            qwen_outputs=qwen_outputs, timings=timings,
            environment=collect_environment(self.cfg, self.qwen if hasattr(self.qwen, "info") else None),
            config=self.cfg.to_dict(), warnings=warnings,
            page_images=[p.color for p in prepared] if keep_page_images else [])

    # ---- Qwen: phân trang, chia chunk, truy hồi ---------------------------
    def _chunks(self, specs: List[FieldSpec]) -> List[List[FieldSpec]]:
        size = max(1, self.cfg.qwen.fields_per_call)
        tables = [s for s in specs if s.is_table]
        scalars = [s for s in specs if not s.is_table]
        chunks = [scalars[i:i + size] for i in range(0, len(scalars), size)]
        chunks += [[t] for t in tables]            # mỗi bảng một lần gọi riêng
        return chunks

    def _call_qwen(self, page_no: int, specs: List[FieldSpec], readings: Dict[str, FieldReadings],
                   prepared: List[PreparedPage], pages_ocr: List[PageOCR], outputs: List[Dict[str, Any]],
                   warnings: List[str]) -> None:
        hints = {s.key: readings[s.key].tesseract.context for s in specs
                 if readings[s.key].tesseract and readings[s.key].tesseract.context}
        t0 = time.perf_counter()
        entry: Dict[str, Any] = {"page": page_no, "fields": [s.key for s in specs], "raw": "", "error": ""}
        try:
            parsed, raw = self.qwen.extract_page(prepared[page_no - 1].color, specs, hints,
                                                 pages_ocr[page_no - 1].text)
            entry["raw"] = raw
            for s in specs:
                val = parsed.get(s.key)
                cur = readings[s.key].qwen
                has_val = val not in (None, "", [], {}) and not (isinstance(val, str) and
                                                                 val.strip().lower() in ("null", "none", "n/a"))
                if has_val and (cur is None or cur.value in (None, "", [])):
                    readings[s.key].qwen = QwenReading(val, page_no, "page", raw)
        except Exception as exc:                    # lỗi 1 lần gọi không làm hỏng tài liệu
            if isinstance(exc, PipelineError):
                raise
            log.exception("Qwen lỗi ở trang %s", page_no)
            entry["error"] = str(exc)
            warnings.append(f"Qwen lỗi ở trang {page_no}: {exc}")
        entry["latency_s"] = round(time.perf_counter() - t0, 2)
        outputs.append(entry)

    def _run_qwen(self, template: Template, readings: Dict[str, FieldReadings], prepared: List[PreparedPage],
                  pages_ocr: List[PageOCR], outputs: List[Dict[str, Any]], warnings: List[str],
                  progress: Optional[ProgressFn]) -> None:
        qcfg = self.cfg.qwen
        n = len(prepared)
        max_calls = qcfg.max_calls_per_doc

        def done(spec: FieldSpec) -> bool:
            q = readings[spec.key].qwen
            return q is not None and q.value not in (None, "", [])

        # Gán trường -> trang. Bằng chứng mạnh nhất: trang nơi Tesseract tìm thấy anchor.
        assigned: Dict[int, List[FieldSpec]] = defaultdict(list)
        unresolved: List[FieldSpec] = []
        for spec in template.fields:
            tr = readings[spec.key].tesseract
            if tr is not None:
                assigned[tr.page].append(spec)
                continue
            hint = spec.page_candidates(n)
            if len(hint) < n:                     # gợi ý trang của template có thông tin
                for p in hint:
                    assigned[p].append(spec)
            else:
                unresolved.append(spec)

        total_est = max(1, sum(len(self._chunks(v)) for v in assigned.values()))
        calls = 0
        # Pass 1: theo trang đã gán
        for page_no in sorted(assigned):
            todo = [s for s in assigned[page_no] if not done(s)]
            for chunk in self._chunks(todo):
                if calls >= max_calls:
                    warnings.append(f"Đã chạm trần {max_calls} lần gọi Qwen/tài liệu – một số trường chưa được Qwen đọc.")
                    break
                self._emit(progress, "qwen", calls / total_est,
                           f"Qwen2-VL trang {page_no}: {', '.join(s.key for s in chunk)[:60]}...")
                self._call_qwen(page_no, chunk, readings, prepared, pages_ocr, outputs, warnings)
                calls += 1

        # Pass 2: trường không có bằng chứng vị trí -> thử lần lượt các trang
        if qcfg.search_all_pages:
            for page_no in range(1, n + 1):
                todo = [s for s in unresolved if not done(s)]
                if not todo:
                    break
                for chunk in self._chunks(todo):
                    if calls >= max_calls:
                        warnings.append("Đã chạm trần số lần gọi Qwen khi quét toàn bộ trang.")
                        break
                    self._emit(progress, "qwen", min(0.95, calls / total_est), f"Qwen2-VL quét trang {page_no}...")
                    self._call_qwen(page_no, chunk, readings, prepared, pages_ocr, outputs, warnings)
                    calls += 1
                if calls >= max_calls:
                    break

        # Pass 3: đọc lại vùng cắt quanh anchor cho trường Qwen bỏ sót (độ phân giải hiệu dụng cao hơn)
        if qcfg.region_refine and hasattr(self.qwen, "read_region"):
            region_calls = 0
            for spec in template.fields:
                if region_calls >= qcfg.max_region_calls:
                    break
                tr = readings[spec.key].tesseract
                if spec.is_table or done(spec) or tr is None or tr.bbox is None:
                    continue
                x0, y0, x1, y1 = tr.bbox
                pad = qcfg.region_pad_px
                img = prepared[tr.page - 1].color
                crop = img.crop((max(0, x0 - 4 * pad), max(0, y0 - pad), min(img.width, x1 + 2 * pad),
                                 min(img.height, y1 + pad)))
                t0 = time.perf_counter()
                try:
                    val, raw = self.qwen.read_region(crop, spec)
                except PipelineError:
                    raise
                except Exception as exc:
                    warnings.append(f"Qwen đọc vùng lỗi ở '{spec.key}': {exc}")
                    continue
                region_calls += 1
                outputs.append({"page": tr.page, "fields": [spec.key], "raw": raw, "error": "",
                                "mode": "region", "latency_s": round(time.perf_counter() - t0, 2)})
                if val:
                    readings[spec.key].qwen = QwenReading(val, tr.page, "region", raw)
        self._emit(progress, "qwen", 1.0, "Hoàn tất Qwen2-VL")
