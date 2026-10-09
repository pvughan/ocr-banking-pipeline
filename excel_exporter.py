"""
excel_exporter.py
=================
Mục 4.3 đề bài – xuất Excel kết quả.

Workbook gồm:
    Ket_qua          9 cột chuẩn: field_key, field_name, value, confidence, source, page,
                     raw_tesseract, raw_qwen, validation_status
    table_<key>      mỗi trường type=table -> một sheet phụ chứa các hàng/cột đã dựng lại
    full_text        văn bản thô từng trang của Tesseract + đầu ra thô (JSON) của Qwen (audit)
    metadata         tên file, template, thời gian xử lý (Tesseract/Qwen/Total), cấu hình, phần cứng
    fusion_audit     (thêm) chiến lược + bằng chứng chọn kết quả, bbox, cờ review (đề bài chấm
                     "có đối chiếu/bằng chứng lựa chọn" – có thể tắt bằng include_audit=False)

Ngoài ra: ``append_trace_log`` ghi nhật ký truy vết JSONL (đầu vào, kết quả, confidence, phiên bản model).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from fusion import OUTPUT_COLUMNS, FieldResult
from pipeline_engine import ExtractionRun
from template_parser import Template

INDIGO = "4F46E5"
EXCEL_CELL_LIMIT = 32000
_THIN = Side(style="thin", color="CBD5E1")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_HEADER_FILL = PatternFill("solid", start_color=INDIGO, end_color=INDIGO)
_HEADER_FONT = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
_STATUS_FILL = {
    "ok": PatternFill("solid", start_color="DCFCE7", end_color="DCFCE7"),
    "warning": PatternFill("solid", start_color="FEF3C7", end_color="FEF3C7"),
    "error": PatternFill("solid", start_color="FEE2E2", end_color="FEE2E2"),
    "missing": PatternFill("solid", start_color="E5E7EB", end_color="E5E7EB"),
}
_REVIEW_FILL = PatternFill("solid", start_color="FFEDD5", end_color="FFEDD5")


# --------------------------------------------------------------------------- #
# Tiện ích ghi ô an toàn
# --------------------------------------------------------------------------- #
def _clean(value: Any) -> Any:
    """Loại ký tự điều khiển bất hợp lệ của XML, cắt quá giới hạn ô Excel, tránh bị hiểu thành công thức."""
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        if len(value) > EXCEL_CELL_LIMIT:
            value = value[:EXCEL_CELL_LIMIT] + " …[đã cắt]"
    return value


def _write_cell(ws, row: int, col: int, value: Any):
    cell = ws.cell(row=row, column=col)
    cell.value = _clean(value)
    if isinstance(cell.value, str) and cell.value[:1] in ("=", "+", "-", "@"):
        cell.data_type = "s"          # giữ nguyên là chuỗi (vd. giá trị OCR bắt đầu bằng '=')
    cell.border = _BORDER
    cell.alignment = Alignment(vertical="top", wrap_text=True)
    return cell


def _write_table(ws, headers: List[str], rows: Iterable[Iterable[Any]]) -> None:
    for j, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=j, value=h)
        c.fill, c.font, c.border = _HEADER_FILL, _HEADER_FONT, _BORDER
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, row in enumerate(rows, start=2):
        for j, v in enumerate(row, start=1):
            _write_cell(ws, i, j, v)
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 24


def _autofit(ws, max_width: int = 60, min_width: int = 10) -> None:
    for j, col in enumerate(ws.columns, start=1):
        longest = 0
        for c in col:
            if c.value is None:
                continue
            longest = max(longest, max(len(part) for part in str(c.value).split("\n")))
        ws.column_dimensions[get_column_letter(j)].width = max(min_width, min(max_width, longest + 2))


def _sheet_name(base: str, used: set) -> str:
    name = re.sub(r"[\[\]:*?/\\]", "_", base)[:31] or "sheet"
    candidate, n = name, 1
    while candidate.lower() in used:
        n += 1
        suffix = f"_{n}"
        candidate = name[:31 - len(suffix)] + suffix
    used.add(candidate.lower())
    return candidate


# --------------------------------------------------------------------------- #
# Các sheet
# --------------------------------------------------------------------------- #
def _sheet_results(wb: Workbook, results: List[FieldResult], used: set) -> None:
    ws = wb.active
    ws.title = _sheet_name("Ket_qua", used)
    rows = [[r.to_row()[c] for c in OUTPUT_COLUMNS] for r in results]
    _write_table(ws, OUTPUT_COLUMNS, rows)
    conf_col = OUTPUT_COLUMNS.index("confidence") + 1
    status_col = OUTPUT_COLUMNS.index("validation_status") + 1
    value_col = OUTPUT_COLUMNS.index("value") + 1
    for i, r in enumerate(results, start=2):
        ws.cell(row=i, column=status_col).fill = _STATUS_FILL.get(r.validation_status, _STATUS_FILL["missing"])
        ws.cell(row=i, column=conf_col).number_format = "0.000"
        if r.needs_review:                    # đánh dấu trường nghi ngờ để human-in-the-loop xem lại
            ws.cell(row=i, column=value_col).fill = _REVIEW_FILL
            ws.cell(row=i, column=value_col).font = Font(bold=True, color="9A3412")
    _autofit(ws)
    ws.column_dimensions[get_column_letter(value_col)].width = 48
    ws.auto_filter.ref = ws.dimensions


def _sheet_tables(wb: Workbook, results: List[FieldResult], used: set) -> None:
    for r in results:
        if r.field_type != "table":
            continue
        ws = wb.create_sheet(_sheet_name(f"table_{r.field_key}", used))
        cols = r.table_columns or (list(r.table_rows[0].keys()) if r.table_rows else [])
        headers = ["row"] + cols
        rows = [[i] + [row.get(c, "") for c in cols] for i, row in enumerate(r.table_rows or [], start=1)]
        _write_table(ws, headers, rows)
        _autofit(ws)


def _sheet_full_text(wb: Workbook, run: ExtractionRun, used: set) -> None:
    ws = wb.create_sheet(_sheet_name("full_text", used))
    rows: List[List[Any]] = []
    for p in run.pages:
        rows.append([p.page_no, "tesseract", f"psm={p.psm}; lang={p.lang}; mean_conf={p.mean_conf:.3f}", p.text])
    for out in run.qwen_outputs:
        info = f"fields={','.join(out.get('fields', []))}; latency_s={out.get('latency_s', '')}"
        if out.get("mode") == "region":
            info += "; mode=region"
        if out.get("error"):
            info += f"; ERROR={out['error']}"
        rows.append([out.get("page"), "qwen_raw_json", info, out.get("raw", "")])
    rows.sort(key=lambda r: (r[0] if isinstance(r[0], int) else 10 ** 6, r[1]))
    _write_table(ws, ["page", "engine", "info", "text"], rows)
    ws.column_dimensions["A"].width = 7
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 42
    ws.column_dimensions["D"].width = 110


def _sheet_metadata(wb: Workbook, run: ExtractionRun, template: Template, results: List[FieldResult],
                    used: set) -> None:
    ws = wb.create_sheet(_sheet_name("metadata", used))
    t = run.timings
    env = run.environment
    cfg = run.config
    counts = {s: sum(1 for r in results if r.validation_status == s) for s in ("ok", "warning", "error", "missing")}
    fusion_s = float(t.get("fusion_s", 0.0))
    total_s = float(t.get("pipeline_s", 0.0)) + fusion_s
    qwen = env.get("qwen", {}) if isinstance(env.get("qwen"), dict) else {}
    rows: List[List[Any]] = [
        ["## Tài liệu", ""],
        ["document_name", run.document_name],
        ["template", run.template_name],
        ["template_fields", len(template.fields)],
        ["pages", run.n_pages],
        ["processed_at", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["## Thời gian xử lý (giây)", ""],
        ["render_s", round(float(t.get("render_s", 0)), 2)],
        ["preprocess_s", round(float(t.get("preprocess_s", 0)), 2)],
        ["Tesseract latency (s)", round(float(t.get("tesseract_s", 0)), 2)],
        ["Qwen latency (s)", round(float(t.get("qwen_s", 0)), 2)],
        ["qwen_model_load_s (một lần/phiên)", round(float(t.get("qwen_load_s", 0)), 2)],
        ["qwen_calls", t.get("qwen_calls", 0)],
        ["fusion_s", round(fusion_s, 3)],
        ["Total latency (s)", round(total_s, 2)],
        ["latency_per_page_s", round(total_s / max(run.n_pages, 1), 2)],
        ["tesseract_per_page_s", ", ".join(str(x) for x in t.get("tesseract_per_page_s", []))],
        ["## Phần cứng / môi trường", ""],
        ["mode", env.get("mode", "CPU only")],
        ["platform", env.get("platform", "")],
        ["machine", env.get("machine", "")],
        ["cpu_count", env.get("cpu_count", "")],
        ["python", env.get("python", "")],
        ["torch", env.get("torch", "")],
        ["transformers", env.get("transformers", "")],
        ["tesseract", f"{env.get('tesseract', '')} (lang={env.get('tesseract_lang', '')})"],
        ["## Cấu hình pipeline", ""],
        ["pipeline_version", env.get("pipeline_version", "")],
        ["prompt_version", env.get("prompt_version", "")],
        ["qwen_model", qwen.get("model_id", cfg.get("qwen", {}).get("model_id", ""))],
        ["qwen_dtype/device", f"{qwen.get('dtype', 'float32')}/{qwen.get('device', 'cpu')}"],
        ["qwen_min_pixels", cfg.get("qwen", {}).get("min_pixels", "")],
        ["qwen_max_pixels", cfg.get("qwen", {}).get("max_pixels", "")],
        ["render_dpi", cfg.get("render", {}).get("dpi", "")],
        ["tesseract_psm_list", str(cfg.get("tesseract", {}).get("psm_list", ""))],
        ["fuzzy_threshold", cfg.get("fusion", {}).get("fuzzy_threshold", "")],
        ["tess_trust_conf", cfg.get("fusion", {}).get("tess_trust_conf", "")],
        ["preprocess", json.dumps(cfg.get("preprocess", {}), ensure_ascii=False)],
        ["## Thống kê kết quả", ""],
        *[[f"fields_{k}", v] for k, v in counts.items()],
        ["fields_need_review", sum(1 for r in results if r.needs_review)],
        ["required_missing", sum(1 for r in results if r.required and r.validation_status == "missing")],
        ["## Cảnh báo", ""],
        *[["warning", w] for w in run.warnings],
    ]
    _write_table(ws, ["key", "value"], rows)
    for i, row in enumerate(rows, start=2):
        if str(row[0]).startswith("##"):
            for j in (1, 2):
                c = ws.cell(row=i, column=j)
                c.font = Font(bold=True, color=INDIGO)
                c.fill = PatternFill("solid", start_color="EEF2FF", end_color="EEF2FF")
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 100


def _sheet_audit(wb: Workbook, results: List[FieldResult], used: set) -> None:
    ws = wb.create_sheet(_sheet_name("fusion_audit", used))
    headers = ["field_key", "type", "required", "strategy", "needs_review", "bbox", "evidence"]
    rows = [[r.field_key, r.field_type, "Y" if r.required else "N", r.strategy, "Y" if r.needs_review else "N",
             "" if not r.bbox else ",".join(str(int(v)) for v in r.bbox), r.evidence] for r in results]
    _write_table(ws, headers, rows)
    _autofit(ws, max_width=40)
    ws.column_dimensions["G"].width = 110


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def build_workbook(results: List[FieldResult], run: ExtractionRun, template: Template,
                   include_audit: bool = True) -> Workbook:
    wb = Workbook()
    used: set = set()
    _sheet_results(wb, results, used)
    _sheet_tables(wb, results, used)
    _sheet_full_text(wb, run, used)
    _sheet_metadata(wb, run, template, results, used)
    if include_audit:
        _sheet_audit(wb, results, used)
    return wb


def export_to_bytes(results: List[FieldResult], run: ExtractionRun, template: Template,
                    include_audit: bool = True) -> bytes:
    """Dùng cho st.download_button."""
    buf = io.BytesIO()
    build_workbook(results, run, template, include_audit).save(buf)
    return buf.getvalue()


def export_to_file(path: Union[str, Path], results: List[FieldResult], run: ExtractionRun, template: Template,
                   include_audit: bool = True) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(export_to_bytes(results, run, template, include_audit))
    return p


def results_to_json(results: List[FieldResult], run: ExtractionRun) -> str:
    """JSON {field_key: {value, confidence, page, bbox, ...}} – dạng mà BPMS yêu cầu (file 'Kết quả cần trả về')."""
    payload: Dict[str, Any] = {
        "document": run.document_name, "template": run.template_name,
        "pipeline_version": run.environment.get("pipeline_version"),
        "prompt_version": run.environment.get("prompt_version"),
        "fields": {},
    }
    for r in results:
        item: Dict[str, Any] = {"value": r.value, "confidence": round(r.confidence, 3), "source": r.source,
                                "page": r.page, "bbox": list(r.bbox) if r.bbox else None,
                                "validation_status": r.validation_status, "needs_review": r.needs_review}
        if r.table_rows is not None:
            item["rows"] = r.table_rows
        payload["fields"][r.field_key] = item
    return json.dumps(payload, ensure_ascii=False, indent=2)


def append_trace_log(path: Union[str, Path], results: List[FieldResult], run: ExtractionRun) -> None:
    """Nhật ký truy vết (JSONL): đầu vào, kết quả, confidence, phiên bản mô hình tại thời điểm xử lý."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    qwen = run.environment.get("qwen", {}) if isinstance(run.environment.get("qwen"), dict) else {}
    record = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "document": run.document_name, "template": run.template_name, "pages": run.n_pages,
        "pipeline_version": run.environment.get("pipeline_version"),
        "prompt_version": run.environment.get("prompt_version"),
        "models": {"tesseract": run.environment.get("tesseract"), "qwen": qwen.get("model_id")},
        "timings_s": {k: v for k, v in run.timings.items() if isinstance(v, (int, float))},
        "results": {r.field_key: {"value": r.value, "confidence": round(r.confidence, 3), "source": r.source,
                                  "status": r.validation_status} for r in results},
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
