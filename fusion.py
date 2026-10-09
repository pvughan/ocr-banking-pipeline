"""
fusion.py
=========
Mục 5.5 đề bài – hợp nhất kết quả Tesseract và Qwen2-VL (trọng tâm chấm điểm).

Hai chiến lược được cài đặt (1 bắt buộc + 1 tuỳ chọn, ở đây làm cả hai và kết hợp):

CHIẾN LƯỢC 1 – ĐỒNG THUẬN (consensus, bắt buộc)
    Chuẩn hoá CẢ HAI giá trị theo type của trường (ngày -> dd/MM/yyyy, tiền -> số nguyên, tên ->
    proper-case...) RỒI mới so sánh bằng ``comparison_key`` (bỏ khoảng trắng thừa, lowercase). Tránh
    "bẫy" normalize sai làm consensus luôn trượt (đề bài mục 13).
      * Khớp 100%                -> source=fused, status=ok, confidence 0.97-1.00
      * Khớp mờ (fuzzy >= 0.90)  -> source=fused, status=ok, confidence 0.95-0.97, đánh dấu review.
        (Trường số/ngày/tiền/mã chỉ chấp nhận khớp CHÍNH XÁC: lệch 1 chữ số vẫn đạt 91% fuzzy!)

CHIẾN LƯỢC 2 – TRỌNG TÀI + CONFIDENCE-WEIGHTED (khi hai engine mâu thuẫn)
    R1  Trường có quy tắc validation/kiểu số-ngày-tiền, Tesseract qua validator VÀ conf từ >= 85%
        -> chọn Tesseract (engine mạnh với ký tự in/mã số).
    R2  Chỉ một bên qua validator -> chọn bên đó.
    R3  Cùng qua hoặc cùng trượt validator -> điểm có trọng số:
            score = prior(loại trường, engine) + conf/grounding + (anchor match)
        * Tesseract: conf trung bình các từ trong TSV.
        * Qwen không có confidence gốc -> dùng GROUNDING: giá trị Qwen có xuất hiện trong văn bản
          Tesseract của trang không? (chống ảo giác: bịa số, bỏ sót trường).
        Mọi quyết định đều ghi `evidence` (đưa vào sheet fusion_audit để chứng minh lựa chọn).

NGUỒN ĐƠN (single-source + validation)
    Chỉ một engine đọc được: qua validator -> nhận với status=warning, ngược lại status=error;
    không engine nào -> missing (trường required cũng được cảnh báo, không bỏ qua âm thầm).

HẬU KIỂM CHÉO
    Số tiền "bằng số" ↔ "bằng chữ" (đề bài 5.6/B4): đọc số bằng chữ tiếng Việt, so với số bằng số.
    Bảng: đối chiếu số hàng giữa Qwen (cấu trúc) và TSV Tesseract.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from config import FusionConfig
from pipeline_engine import ExtractionRun, FieldReadings
from template_parser import (ColumnSpec, FieldSpec, Template, ValidationResult, clean_whitespace, comparison_key,
                             fold_vietnamese, parse_money_to_int, postprocess_value, validate_value, words_to_int)

log = logging.getLogger("ocr.fusion")

SOURCES = ("tesseract", "qwen", "fused", "manual")
STATUSES = ("ok", "warning", "error", "missing")
OUTPUT_COLUMNS = ["field_key", "field_name", "value", "confidence", "source", "page",
                  "raw_tesseract", "raw_qwen", "validation_status"]


# --------------------------------------------------------------------------- #
# Kết quả
# --------------------------------------------------------------------------- #
@dataclass
class FieldResult:
    field_key: str
    field_name: str
    value: str
    confidence: float
    source: str                     # tesseract | qwen | fused | manual
    page: Optional[int]
    raw_tesseract: str
    raw_qwen: str
    validation_status: str          # ok | warning | error | missing
    # --- thông tin phụ (không nằm trong 9 cột chuẩn) ---
    strategy: str = ""
    evidence: str = ""
    needs_review: bool = False
    field_type: str = "text"
    required: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    table_rows: Optional[List[Dict[str, str]]] = None
    table_columns: Optional[List[str]] = None

    def to_row(self) -> Dict[str, Any]:
        return {"field_key": self.field_key, "field_name": self.field_name, "value": self.value,
                "confidence": round(float(self.confidence), 3), "source": self.source,
                "page": self.page if self.page is not None else "", "raw_tesseract": self.raw_tesseract,
                "raw_qwen": self.raw_qwen, "validation_status": self.validation_status}


@dataclass
class _Candidate:
    engine: str                     # tesseract | qwen
    raw: str
    norm: str
    key: str                        # comparison_key
    validation: ValidationResult
    conf: Optional[float]           # conf Tesseract (0-1); None với Qwen
    page: Optional[int]
    bbox: Optional[Tuple[int, int, int, int]]
    grounding: float                # độ "có căn cứ" trong văn bản OCR (0-1)
    anchor_score: float = 0.0
    changes: List[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return self.validation.status != "error"

    @property
    def valid_ok(self) -> bool:
        return self.validation.status == "ok"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _qwen_to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    s = str(value).strip()
    return s if s.lower() not in ("null", "none") else None


class _PagePool:
    """Văn bản OCR đã 'fold' của từng trang -> kiểm tra grounding của giá trị Qwen."""

    def __init__(self, run: ExtractionRun):
        self.folded = {p.page_no: fold_vietnamese(p.text) for p in run.pages}
        self.digits = {p.page_no: re.sub(r"\D", "", p.text) for p in run.pages}

    def grounding(self, spec: Any, norm: str, page: Optional[int]) -> float:
        if spec.type in ("checkbox", "boolean") or not norm:
            return 0.5                                   # không kiểm chứng được bằng văn bản
        pages = [page] if page in self.folded else []
        pages += [p for p in self.folded if p not in pages]
        digits = re.sub(r"\D", "", norm)
        digit_like = spec.type in ("money", "number", "date") or (len(digits) >= 4 and len(digits) >= 0.6 * len(norm))
        best = 0.0
        if digit_like and len(digits) >= 3:
            for p in pages:
                d = self.digits[p]
                if digits in d:
                    return 1.0
                if d:   # chỉ cho điểm khi gần như khớp (lệch ~1 chữ số); số ngẫu nhiên vẫn ~55% partial_ratio
                    pr = fuzz.partial_ratio(digits, d) / 100.0
                    best = max(best, 0.8 * max(0.0, (pr - 0.80) / 0.20))
            return best
        f = fold_vietnamese(norm)
        if len(f) < 3:
            return 0.5
        for p in pages:
            best = max(best, fuzz.partial_ratio(f, self.folded[p]) / 100.0)
            if best >= 0.95:
                break
        return best


# --------------------------------------------------------------------------- #
# Engine hợp nhất
# --------------------------------------------------------------------------- #
class FusionEngine:
    def __init__(self, cfg: Optional[FusionConfig] = None):
        self.cfg = cfg or FusionConfig()

    # ------------------------------------------------------------------ #
    def fuse_document(self, template: Template, run: ExtractionRun) -> List[FieldResult]:
        pool = _PagePool(run)
        results: List[FieldResult] = []
        for spec in template.fields:
            rd = run.readings.get(spec.key) or FieldReadings()
            try:
                res = self._fuse_table(spec, rd) if spec.is_table else self._fuse_scalar(spec, rd, pool)
            except Exception as exc:                          # một trường lỗi không làm hỏng cả bản ghi
                log.exception("Fusion lỗi ở trường %s", spec.key)
                res = self._result(spec, "", 0.0, "fused", None, "", "", "error", "fusion_exception",
                                   f"Lỗi nội bộ khi fusion: {exc}", True)
            results.append(res)
        self._cross_check_amounts(results, template)
        return results

    # ------------------------------------------------------------------ #
    def _result(self, spec: FieldSpec, value: str, conf: float, source: str, page: Optional[int], raw_t: str,
                raw_q: str, status: str, strategy: str, evidence: str, review: bool,
                bbox: Optional[Tuple[int, int, int, int]] = None) -> FieldResult:
        return FieldResult(spec.field_key, spec.field_name, value, round(_clamp(conf, 0.0, 1.0), 3), source, page,
                           raw_t, raw_q, status, strategy, evidence, review, spec.type, spec.required, bbox)

    def _candidate(self, engine: str, spec: FieldSpec, raw: Optional[str], conf: Optional[float],
                   page: Optional[int], bbox: Optional[Tuple[int, int, int, int]], anchor: float,
                   pool: _PagePool) -> Optional[_Candidate]:
        if raw is None or not clean_whitespace(raw):
            return None
        pp = postprocess_value(spec, raw, keep_verbatim=False)       # luôn chuẩn hoá TRƯỚC khi so sánh
        val = validate_value(spec.validation, pp.value, spec.type)
        if not pp.ok and val.status == "ok":
            val = ValidationResult("error", "; ".join(pp.changes) or "không chuẩn hoá được theo kiểu dữ liệu")
        return _Candidate(engine, clean_whitespace(raw), pp.value, comparison_key(spec.type, pp.value, spec.enum_values),
                          val, conf, page, bbox, pool.grounding(spec, pp.value, page), anchor, pp.changes)

    # ---- trọng số theo loại trường --------------------------------------
    @staticmethod
    def _priors(spec: FieldSpec) -> Tuple[float, float]:
        """(prior Tesseract, prior Qwen). Mã số/ngày/tiền -> Tesseract; địa chỉ/checkbox/văn bản dài -> Qwen."""
        if spec.type in ("checkbox", "boolean") or spec.enum_values:
            return 0.15, 0.85
        if spec.type in ("money", "number", "date"):
            return 0.60, 0.40
        val = spec.validation.lower()
        if any(t in val for t in ("msdn", "mst", "cccd", "cmnd", "phone", "email", "^")):
            return 0.60, 0.40
        long_text = any(t in spec.field_key for t in ("dia_chi", "noi_dung", "ghi_chu", "ten_", "nguoi_")) \
            or "địa chỉ" in spec.field_name.lower()
        return (0.40, 0.60) if long_text else (0.50, 0.50)

    @staticmethod
    def _exact_only(spec: FieldSpec, key: str) -> bool:
        """Trường số/ngày/tiền/mã: KHÔNG chấp nhận khớp mờ (lệch 1 chữ số vẫn đạt ~91% fuzzy)."""
        if spec.type in ("money", "number", "date"):
            return True
        digits = sum(c.isdigit() for c in key)
        return bool(spec.validation) and digits >= 0.5 * max(len(key), 1)

    def _final_value(self, cand: _Candidate) -> str:
        return cand.raw if self.cfg.keep_verbatim_value else cand.norm

    # ---- trường đơn -------------------------------------------------------
    def _fuse_scalar(self, spec: FieldSpec, rd: FieldReadings, pool: _PagePool) -> FieldResult:
        cfg = self.cfg
        tr, qr = rd.tesseract, rd.qwen
        raw_t = tr.value if tr and tr.value else ""
        q_str = _qwen_to_str(qr.value) if qr else None
        raw_q = q_str or ""
        t = self._candidate("tesseract", spec, tr.value if tr else None, tr.confidence if tr else None,
                            tr.page if tr else None, tr.bbox if tr else None, tr.anchor_score if tr else 0.0, pool)
        q = self._candidate("qwen", spec, q_str, None, qr.page if qr else None, None, 0.0, pool)

        # ---- không engine nào đọc được ----
        if t is None and q is None:
            ev = "Cả Tesseract và Qwen đều không trích được giá trị."
            if spec.required:
                ev += " Trường BẮT BUỘC (required=Y) bị thiếu -> cần cán bộ nhập tay."
            return self._result(spec, "", 0.0, "fused", None, raw_t, raw_q, "missing", "both_missing", ev,
                                spec.required)

        # ---- nguồn đơn + validator ----
        if t is None or q is None:
            c = t or q
            assert c is not None
            if c.engine == "tesseract":
                conf = min(cfg.single_source_cap, 0.50 + 0.30 * (c.conf or 0.0))
                note = f"Chỉ Tesseract đọc được (conf từ {c.conf:.2f}); Qwen không trả giá trị."
            else:
                conf = min(cfg.single_source_cap, 0.45 + 0.25 * c.grounding)
                note = f"Chỉ Qwen đọc được; grounding trong OCR text = {c.grounding:.2f}."
                if c.grounding < 0.4 and self._exact_only(spec, c.key):
                    conf = min(conf, cfg.ungrounded_qwen_cap)
                    note += " Giá trị KHÔNG xuất hiện trong văn bản Tesseract -> nghi ảo giác."
            if c.passes:
                status = "warning"          # đạt validator nhưng chỉ có một nguồn -> cảnh báo
                note += " Qua validator -> nhận kèm cảnh báo."
                if c.validation.status == "warning":
                    note += f" ({c.validation.message})"
            else:
                status = "error"
                conf = min(conf, 0.30)
                note += f" TRƯỢT validator: {c.validation.message}"
            return self._result(spec, self._final_value(c), conf, c.engine, c.page, raw_t, raw_q, status,
                                "single_source_validation", note, True, c.bbox)

        # ---- hai nguồn: CHIẾN LƯỢC 1 – đồng thuận ----
        page = t.page or q.page
        if t.key == q.key and t.key:
            base = cfg.exact_conf_range[0] + (cfg.exact_conf_range[1] - cfg.exact_conf_range[0]) * (t.conf or 0.5)
            if t.passes:
                return self._result(spec, self._final_value(t), base, "fused", page, raw_t, raw_q, "ok",
                                    "consensus_exact", "Hai engine khớp 100% sau chuẩn hoá.", False, t.bbox)
            return self._result(spec, self._final_value(t), min(base, 0.60), "fused", page, raw_t, raw_q, "error",
                                "consensus_exact_invalid",
                                f"Hai engine khớp nhau nhưng TRƯỢT validator: {t.validation.message}", True, t.bbox)

        # Tesseract hay rơi dấu tiếng Việt: hai bản chỉ khác dấu -> đồng thuận, chọn bản CÓ dấu.
        if t.key and fold_vietnamese(t.norm) == fold_vietnamese(q.norm):
            def n_marks(c: _Candidate) -> int:
                return sum(1 for ch in c.norm if ord(ch) > 127)
            pick = q if n_marks(q) >= n_marks(t) else t
            if t.passes != q.passes:
                pick = t if t.passes else q
            lo, hi = cfg.fuzzy_conf_range
            return self._result(spec, self._final_value(pick), hi, "fused", page, raw_t, raw_q,
                                "ok" if pick.passes else "error", "consensus_diacritics",
                                f"Hai engine chỉ khác dấu tiếng Việt; chọn bản của {pick.engine}. Gắn cờ review.",
                                True, t.bbox)

        ratio = fuzz.ratio(t.key, q.key) / 100.0
        if ratio >= cfg.fuzzy_threshold and not self._exact_only(spec, t.key):
            lo, hi = cfg.fuzzy_conf_range
            conf = lo + (hi - lo) * (ratio - cfg.fuzzy_threshold) / max(1e-9, 1.0 - cfg.fuzzy_threshold)
            pick = self._pick_for_fuzzy(spec, t, q)
            status = "ok" if pick.passes else "error"
            return self._result(spec, self._final_value(pick), conf if pick.passes else min(conf, 0.6), "fused", page,
                                raw_t, raw_q, status, "consensus_fuzzy",
                                f"Khớp mờ {ratio:.2f} >= {cfg.fuzzy_threshold:.2f}; chọn bản của {pick.engine} "
                                f"(qua validator={pick.passes}). Gắn cờ review.", True, t.bbox)

        # ---- CHIẾN LƯỢC 2 – trọng tài / confidence-weighted ----
        return self._arbitrate(spec, t, q, raw_t, raw_q, ratio)

    def _pick_for_fuzzy(self, spec: FieldSpec, t: _Candidate, q: _Candidate) -> _Candidate:
        if t.passes != q.passes:
            return t if t.passes else q
        pt, pq = self._priors(spec)
        return t if pt > pq else q

    def _arbitrate(self, spec: FieldSpec, t: _Candidate, q: _Candidate, raw_t: str, raw_q: str,
                   ratio: float) -> FieldResult:
        cfg = self.cfg
        pt, pq = self._priors(spec)
        has_rule = bool(spec.validation) or spec.type in ("date", "money", "number")
        t_conf = t.conf or 0.0
        head = (f"Mâu thuẫn (độ giống {ratio:.2f}): tesseract='{t.norm[:50]}' (conf {t_conf:.2f}, "
                f"validator={t.validation.status}) vs qwen='{q.norm[:50]}' (validator={q.validation.status}, "
                f"grounding {q.grounding:.2f}).")
        page = t.page or q.page

        # R1: Tesseract tin cậy cao + qua validator
        if has_rule and t.valid_ok and t_conf >= cfg.tess_trust_conf and (pt >= pq or not q.passes):
            conf = 0.80 + 0.10 * _clamp((t_conf - cfg.tess_trust_conf) / max(1e-9, 1 - cfg.tess_trust_conf), 0, 1)
            both_valid = q.valid_ok       # hai giá trị KHÁC NHAU đều hợp lệ = xung đột thật -> cần người xem
            note = f" R1: Tesseract conf >= {cfg.tess_trust_conf:.2f} và qua validator -> chọn Tesseract."
            if both_valid:
                note += " Lưu ý: giá trị của Qwen cũng hợp lệ -> gắn cờ review."
            return self._result(spec, self._final_value(t), conf, "tesseract", t.page or page, raw_t, raw_q, "ok",
                                "arbiter_R1_tesseract_high_conf", head + note,
                                both_valid or conf < cfg.review_threshold, t.bbox)

        # R2: chỉ một bên qua validator
        if t.passes != q.passes:
            win = t if t.passes else q
            conf = 0.70 + 0.10 * (win.conf if win.engine == "tesseract" else win.grounding)
            status = "ok" if win.valid_ok else "warning"
            if win.engine == "qwen" and win.grounding < 0.4:
                conf, status = min(conf, cfg.ungrounded_qwen_cap + 0.1), "warning"
            return self._result(spec, self._final_value(win), conf, win.engine, win.page or page, raw_t, raw_q,
                                status, "arbiter_R2_validator",
                                head + f" R2: chỉ {win.engine} qua validator -> chọn {win.engine}.",
                                conf < cfg.review_threshold, t.bbox if win is t else None)

        # R3: trọng số confidence + grounding + prior theo loại trường
        s_t = pt + 0.35 * (t_conf - 0.5) + 0.15 * t.anchor_score
        s_q = pq + 0.35 * (q.grounding - 0.5)
        win, lose, s_w, s_l = (t, q, s_t, s_q) if s_t >= s_q else (q, t, s_q, s_t)
        margin = s_w - s_l
        conf = _clamp(0.55 + 0.9 * margin, 0.45, 0.90)
        if not win.passes:                                # cả hai cùng trượt validator
            conf, status = min(conf, 0.40), "error"
        else:
            status = "ok" if margin >= cfg.arbitration_margin_warning else "warning"
        return self._result(
            spec, self._final_value(win), conf, win.engine, win.page or page, raw_t, raw_q, status,
            "arbiter_R3_weighted",
            head + f" R3: score tesseract={s_t:.2f} vs qwen={s_q:.2f} (prior {pt:.2f}/{pq:.2f}) -> chọn "
                   f"{win.engine}, chênh {margin:.2f}.", True, t.bbox if win is t else None)

    # ---- bảng -------------------------------------------------------------
    @staticmethod
    def _rows_from_qwen(spec: FieldSpec, value: Any) -> List[Dict[str, str]]:
        rows = value
        if isinstance(rows, str):
            try:
                rows = json.loads(rows)
            except json.JSONDecodeError:
                return []
        if isinstance(rows, dict):
            rows = next((v for v in rows.values() if isinstance(v, list)), [rows])
        if not isinstance(rows, list):
            return []
        keys = [c.column_key for c in spec.columns]
        out: List[Dict[str, str]] = []
        for r in rows:
            if isinstance(r, dict):
                if not spec.columns:
                    row = {str(k): "" if v is None else str(v) for k, v in r.items()}
                else:
                    folded = {fold_vietnamese(str(k)): v for k, v in r.items()}
                    row = {}
                    for c in spec.columns:
                        v = r.get(c.column_key)
                        if v is None:
                            v = folded.get(fold_vietnamese(c.column_key), folded.get(fold_vietnamese(c.column_name)))
                        row[c.column_key] = "" if v is None else str(v)
            elif isinstance(r, (list, tuple)) and keys:
                row = {k: "" if i >= len(r) or r[i] is None else str(r[i]) for i, k in enumerate(keys)}
            else:
                continue
            if any(clean_whitespace(v) for v in row.values()):
                out.append(row)
        return out

    @staticmethod
    def _rows_from_tess(spec: FieldSpec, rows: Optional[List[List[str]]]) -> List[Optional[Dict[str, str]]]:
        """Chỉ ánh xạ được khi số ô == số cột (nếu không -> None, vẫn giữ để đếm hàng)."""
        keys = [c.column_key for c in spec.columns]
        out: List[Optional[Dict[str, str]]] = []
        for r in rows or []:
            out.append({k: r[i] for i, k in enumerate(keys)} if keys and len(r) == len(keys) else None)
        return out

    def _fuse_table(self, spec: FieldSpec, rd: FieldReadings) -> FieldResult:
        cfg = self.cfg
        tr, qr = rd.tesseract, rd.qwen
        q_rows = self._rows_from_qwen(spec, qr.value) if qr else []
        t_rows_raw = tr.rows if tr else None
        t_rows = self._rows_from_tess(spec, t_rows_raw)
        raw_t = json.dumps(t_rows_raw, ensure_ascii=False) if t_rows_raw else ""
        raw_q = json.dumps(q_rows, ensure_ascii=False) if q_rows else ""
        cols: List[ColumnSpec] = spec.columns or [ColumnSpec(k) for k in (q_rows[0].keys() if q_rows else [])]
        col_keys = [c.column_key for c in cols]

        if not q_rows and not any(t_rows):
            return self._result(spec, "", 0.0, "fused", None, raw_t, raw_q, "missing", "both_missing",
                                "Không dựng được bảng từ cả hai engine.", spec.required)

        base_rows, source = (q_rows, "qwen") if q_rows else ([r for r in t_rows if r], "tesseract")
        compared = agreed = replaced = bad_cells = 0
        out_rows: List[Dict[str, str]] = []
        for i, row in enumerate(base_rows):
            trow = t_rows[i] if (q_rows and i < len(t_rows)) else None
            new: Dict[str, str] = {}
            for c in cols:
                qn = postprocess_value(c, row.get(c.column_key, "")).value
                qv = validate_value(c.validation, qn, c.type)
                if trow is not None:
                    tn = postprocess_value(c, trow.get(c.column_key, "")).value
                    tv = validate_value(c.validation, tn, c.type)
                    compared += 1
                    if comparison_key(c.type, qn) == comparison_key(c.type, tn):
                        agreed += 1
                    elif not qv.passed and tv.passed and tn:
                        qn, qv, replaced = tn, tv, replaced + 1
                if not qv.passed:
                    bad_cells += 1
                new[c.column_key] = qn
            out_rows.append(new)

        n_t, n_q = len(t_rows_raw or []), len(q_rows)
        count_ok = (n_t == n_q) if (q_rows and t_rows_raw) else None
        if compared:
            conf = 0.55 + 0.40 * agreed / compared
            source = "fused"
        else:
            conf = 0.55 if q_rows else 0.40
        notes = [f"{len(out_rows)} hàng × {len(col_keys)} cột"]
        if compared:
            notes.append(f"đối chiếu {compared} ô: {agreed} khớp, thay {replaced} ô theo Tesseract")
        if count_ok is False:
            conf -= 0.10
            notes.append(f"số hàng lệch: Qwen {n_q} vs Tesseract TSV {n_t}")
        if not t_rows_raw:
            notes.append("không có TSV Tesseract để đối chiếu cấu trúc")
        if bad_cells:
            status, conf = "error", min(conf, 0.50)
            notes.append(f"{bad_cells} ô trượt validator")
        elif count_ok is True and compared and agreed / compared >= 0.8:
            status = "ok"
        else:
            status = "warning"
        page = (qr.page if qr else None) or (tr.page if tr else None)
        res = self._result(spec, f"[bảng {len(out_rows)} hàng × {len(col_keys)} cột] -> sheet table_{spec.key}",
                           conf, source, page, raw_t, raw_q, status, "table_structure_crosscheck",
                           "; ".join(notes), status != "ok" or conf < cfg.review_threshold,
                           tr.bbox if tr else None)
        res.table_rows, res.table_columns = out_rows, col_keys
        return res

    # ---- hậu kiểm chéo: số bằng số ↔ bằng chữ -----------------------------
    def _cross_check_amounts(self, results: List[FieldResult], template: Template) -> None:
        by_key = {r.field_key: r for r in results}
        for r in results:
            counterpart_key = None
            spec = template.get(r.field_key)
            m = re.search(r"pair=([\w.\[\]]+)", spec.notes if spec else "")
            if m:
                counterpart_key = m.group(1)
            elif r.field_key.endswith("_bang_so"):
                counterpart_key = r.field_key[: -len("_bang_so")] + "_bang_chu"
            if not counterpart_key or counterpart_key not in by_key:
                continue
            words = by_key[counterpart_key]
            if not r.value or not words.value:
                continue
            digits_val = parse_money_to_int(r.value)
            words_val = words_to_int(words.value)
            if digits_val is None or words_val is None:
                for x in (r, words):
                    x.evidence += " | Không phân tích được cặp bằng số/bằng chữ để đối chiếu."
                    x.needs_review = True
                continue
            if digits_val == words_val:
                for x in (r, words):
                    x.evidence += " | Bằng số khớp bằng chữ."
                continue
            for x in (r, words):
                x.evidence += (f" | LỆCH: bằng số = {digits_val:,} nhưng bằng chữ = {words_val:,}. "
                               f"Cần cán bộ kiểm tra (không tự chọn bên nào).")
                x.needs_review = True
                x.confidence = round(max(0.2, x.confidence * 0.7), 3)
                if x.validation_status == "ok":
                    x.validation_status = "warning"


# --------------------------------------------------------------------------- #
# Hỗ trợ human-in-the-loop (app.py)
# --------------------------------------------------------------------------- #
def apply_manual_edit(result: FieldResult, spec: FieldSpec, new_value: str) -> FieldResult:
    """
    Cán bộ sửa tay ô `value`: source -> 'manual', confidence = 1.0, kiểm tra lại validator.
    Giá trị rỗng với trường bắt buộc -> 'missing'; vẫn trượt validator -> 'warning' (con người chịu trách nhiệm).
    """
    value = clean_whitespace(new_value or "")
    if not value:
        status = "missing" if spec.required else "ok"
        return replace(result, value="", source="manual", confidence=1.0, validation_status=status,
                       needs_review=spec.required, strategy="manual_edit", evidence="Cán bộ xoá giá trị.")
    pp = postprocess_value(spec, value)
    val = validate_value(spec.validation, pp.value, spec.type)
    status = "ok" if (val.ok and pp.ok) else "warning"
    ev = "Cán bộ sửa tay." + ("" if status == "ok" else f" Lưu ý: {val.message or 'không chuẩn hoá được'}.")
    return replace(result, value=pp.value, source="manual", confidence=1.0, validation_status=status,
                   needs_review=False, strategy="manual_edit", evidence=ev)
