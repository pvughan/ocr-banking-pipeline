"""
app.py – Giao diện Streamlit cho OCR-PIPELINE-2026 (Tesseract + Qwen2-VL local, CPU, offline).

Chạy:   streamlit run app.py

Luồng 5 bước (phục vụ quay video demo 5-10 phút):
    1. Template   – chọn mẫu có sẵn / upload Excel / tạo-sửa trong app rồi export lại Excel chuẩn
    2. Tài liệu   – upload PDF nhiều trang hoặc ảnh; thumbnail + xem từng trang
    3. Pipeline   – Render -> Tesseract TSV -> Qwen2-VL -> Fusion, thanh tiến trình + thẻ metrics
    4. Review     – bảng 9 cột sửa trực tiếp (st.data_editor); ô sửa -> source='manual'
    5. Xuất Excel – 3 sheet chuẩn (Ket_qua, full_text, metadata) + sheet bảng + fusion_audit
"""
from __future__ import annotations

import io
import json
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from config import LOG_DIR, TEMPLATE_DIR, AppConfig, PROMPT_VERSION, RenderConfig
from excel_exporter import append_trace_log, export_to_bytes, results_to_json
from fusion import OUTPUT_COLUMNS, FieldResult, FusionEngine, apply_manual_edit
from pipeline_engine import ExtractionRun, OCRPipeline, PipelineError, QwenVLEngine, render_document
from template_parser import (TEMPLATE_COLUMNS, VALID_TYPES, Template, TemplateError, parse_template,
                             postprocess_value, template_from_dataframe, template_to_excel_bytes, validate_value)

st.set_page_config(page_title="OCR Ngân hàng · Tesseract + Qwen2-VL", page_icon="🧾", layout="wide")

def _stretch_kwargs() -> dict:
    """Streamlit >= 1.50 dùng width='stretch'; bản cũ dùng use_container_width=True."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


STRETCH = _stretch_kwargs()

STEPS = ["1 · Template", "2 · Tài liệu", "3 · Chạy pipeline", "4 · Review & sửa", "5 · Xuất Excel"]
INDIGO = "#4F46E5"

st.markdown(f"""
<style>
.block-container {{padding-top: 1.4rem; max-width: 1280px;}}
.stepper {{display:flex; gap:8px; margin-bottom:1rem;}}
.step {{flex:1; padding:10px 12px; border-radius:10px; background:#F1F5F9; color:#475569; font-weight:600;
        border:1px solid #E2E8F0; text-align:center; font-size:0.92rem;}}
.step.active {{background:{INDIGO}; color:white; border-color:{INDIGO};}}
.step.done {{background:#E0E7FF; color:#3730A3; border-color:#C7D2FE;}}
.pill {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600;}}
.pill.ok {{background:#DCFCE7; color:#166534;}} .pill.warning {{background:#FEF3C7; color:#92400E;}}
.pill.error {{background:#FEE2E2; color:#991B1B;}} .pill.missing {{background:#E5E7EB; color:#374151;}}
div[data-testid="stMetric"] {{background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:10px 14px;}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Trạng thái phiên
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    defaults = {"step_radio": STEPS[0], "template": None, "doc_bytes": None, "doc_name": "", "doc_sig": None,
                "previews": [], "run": None, "results": None, "run_id": 0, "final": None, "pipeline_error": ""}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def goto(step_idx: int) -> None:
    st.session_state["step_radio"] = STEPS[step_idx]


def step_index() -> int:
    return STEPS.index(st.session_state["step_radio"])


def list_builtin_templates() -> Dict[str, Path]:
    TEMPLATE_DIR.mkdir(exist_ok=True)
    return {p.stem: p for p in sorted(TEMPLATE_DIR.glob("*.xlsx")) if not p.name.startswith("~")}


def draw_stepper() -> None:
    cur = step_index()
    html = "".join(
        f'<div class="step {"active" if i == cur else "done" if i < cur else ""}">{"✅ " if i < cur else ""}{s}</div>'
        for i, s in enumerate(STEPS))
    st.markdown(f'<div class="stepper">{html}</div>', unsafe_allow_html=True)


def pill(status: str) -> str:
    return f'<span class="pill {status}">{status}</span>'


# --------------------------------------------------------------------------- #
# Sidebar: điều hướng + cấu hình
# --------------------------------------------------------------------------- #
def sidebar() -> AppConfig:
    st.sidebar.title("🧾 OCR-PIPELINE-2026")
    st.sidebar.caption("Tesseract + Qwen2-VL · chạy 100% local, CPU")
    st.sidebar.radio("Quy trình", STEPS, key="step_radio")
    cfg = AppConfig()
    with st.sidebar.expander("⚙️ Cấu hình nâng cao"):
        cfg.render.dpi = st.slider("DPI render PDF", 200, 300, 200, 50)
        psm = st.multiselect("Tesseract PSM", [3, 4, 6, 11], default=[6],
                             help="Chọn nhiều PSM -> lấy PSM có confidence cao nhất (chậm hơn)")
        cfg.tesseract.psm_list = tuple(psm or [6])
        cfg.qwen.fields_per_call = st.slider("Số trường / lần gọi Qwen", 2, 12, 6)
        cfg.qwen.max_calls_per_doc = st.slider("Trần số lần gọi Qwen / tài liệu", 5, 120, 40)
        cfg.qwen.region_refine = st.checkbox("Đọc lại vùng cắt khi Qwen bỏ sót", True)
        cfg.preprocess.enabled = st.checkbox("Tiền xử lý OpenCV", True)
        cfg.fusion.keep_verbatim_value = st.checkbox(
            "Giữ giá trị NGUYÊN VĂN (không chuẩn hoá)", False,
            help="Bật nếu bên nhận yêu cầu bóc nguyên văn; việc chuẩn hoá để so khớp vẫn được thực hiện nội bộ.")
        cfg.fusion.fuzzy_threshold = st.slider("Ngưỡng fuzzy consensus", 0.80, 1.00, 0.90, 0.01)
    st.sidebar.caption(f"Prompt: `{PROMPT_VERSION}`")
    return cfg


# --------------------------------------------------------------------------- #
# Bước 1 – Template
# --------------------------------------------------------------------------- #
def step_template() -> None:
    st.subheader("Bước 1 · Quản lý template")
    builtin = list_builtin_templates()
    mode = st.radio("Nguồn template", ["Template có sẵn (B1/B2/B5)", "Upload file Excel", "Tạo / sửa trong ứng dụng"],
                    horizontal=True)

    if mode.startswith("Template có sẵn"):
        if not builtin:
            st.warning(f"Chưa có file template trong thư mục `{TEMPLATE_DIR}`.")
        else:
            name = st.selectbox("Chọn template", list(builtin))
            if st.button("Dùng template này", type="primary"):
                _load_template(lambda: parse_template(builtin[name], name=name))
    elif mode == "Upload file Excel":
        up = st.file_uploader("Template Excel (sheet `template` + tuỳ chọn `table_<field_key>`)", type=["xlsx"])
        if up is not None and st.button("Đọc template", type="primary"):
            _load_template(lambda: parse_template(up, name=Path(up.name).stem))
    else:
        tpl: Optional[Template] = st.session_state.template
        base = tpl.to_dataframe() if tpl else pd.DataFrame(
            [{"field_key": "so_van_ban", "field_name": "Số văn bản", "type": "text", "required": "N",
              "validation": "", "anchor": "Số:", "page": "1", "notes": ""}], columns=TEMPLATE_COLUMNS)
        st.caption("Thêm/sửa/xoá dòng. Trường `table`: khai báo cột con trong notes, ví dụ `columns=stt|ten_ts|so_luong|gia_tri`.")
        edited = st.data_editor(
            base, num_rows="dynamic", **STRETCH, hide_index=True, key="tpl_editor",
            column_config={"type": st.column_config.SelectboxColumn("type", options=list(VALID_TYPES), required=True),
                           "required": st.column_config.SelectboxColumn("required", options=["Y", "N"])})
        c1, c2 = st.columns(2)
        name = c1.text_input("Tên template", value=(tpl.name if tpl else "template_tuy_chinh"))
        if c2.button("Áp dụng template đang soạn", type="primary"):
            _load_template(lambda: template_from_dataframe(edited, name=name))

    tpl = st.session_state.template
    if tpl:
        st.success(f"Template đang dùng: **{tpl.name}** · {len(tpl.fields)} trường "
                   f"({len(tpl.table_fields)} bảng) · bắt buộc: {sum(f.required for f in tpl.fields)}")
        for w in tpl.warnings:
            st.warning(w)
        st.dataframe(tpl.to_dataframe(), **STRETCH, hide_index=True)
        for f in tpl.table_fields:
            with st.expander(f"Cột con của bảng `{f.field_key}`"):
                st.dataframe(pd.DataFrame([asdict(c) for c in f.columns]), hide_index=True, **STRETCH)
        st.download_button("⬇️ Export template (Excel chuẩn)", template_to_excel_bytes(tpl),
                           file_name=f"{tpl.name}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.button("Tiếp tục: tải tài liệu ▶", on_click=goto, args=(1,), type="primary")


def _load_template(loader) -> None:
    try:
        st.session_state.template = loader()
        st.session_state.results = st.session_state.run = st.session_state.final = None
    except TemplateError as exc:
        st.error(f"Template không hợp lệ: {exc}")
    except Exception as exc:
        st.error(f"Lỗi đọc template: {exc}")


# --------------------------------------------------------------------------- #
# Bước 2 – Tài liệu
# --------------------------------------------------------------------------- #
def step_document() -> None:
    st.subheader("Bước 2 · Tải tài liệu")
    if not st.session_state.template:
        st.info("Hãy chọn template ở Bước 1 trước.")
        return
    up = st.file_uploader("PDF (nhiều trang) hoặc ảnh JPG/PNG/TIFF", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
    if up is not None:
        sig = (up.name, up.size)
        if sig != st.session_state.doc_sig:
            data = up.getvalue()
            try:
                with st.spinner("Đang render xem trước..."):
                    pages = render_document(data, up.name, RenderConfig(dpi=80))
            except PipelineError as exc:
                st.error(str(exc))
                return
            st.session_state.update(doc_bytes=data, doc_name=up.name, doc_sig=sig,
                                    previews=[p.image for p in pages], run=None, results=None, final=None)
    previews: List[Image.Image] = st.session_state.previews
    if not previews:
        return
    n = len(previews)
    st.success(f"**{st.session_state.doc_name}** · {n} trang")
    cols = st.columns(min(n, 8))
    for i, im in enumerate(previews):
        with cols[i % len(cols)]:
            st.image(im, caption=f"Trang {i + 1}", **STRETCH)
    page = st.slider("Xem chi tiết trang", 1, n, 1) if n > 1 else 1
    st.image(previews[page - 1], caption=f"Trang {page}/{n}", width=620)
    st.button("Tiếp tục: chạy pipeline ▶", on_click=goto, args=(2,), type="primary")


# --------------------------------------------------------------------------- #
# Bước 3 – Pipeline
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _cached_qwen_engine(model_id: str, min_pixels: int, max_pixels: int, local_only: bool) -> QwenVLEngine:
    """Nạp model MỘT lần cho cả phiên Streamlit (nạp lại mỗi lần bấm nút sẽ rất chậm trên CPU).
    Khoá cache chỉ gồm tham số ảnh hưởng tới việc nạp model."""
    return QwenVLEngine(replace(AppConfig().qwen, model_id=model_id, min_pixels=min_pixels,
                                max_pixels=max_pixels, local_files_only=local_only))


def get_qwen_engine(cfg: AppConfig) -> QwenVLEngine:
    engine = _cached_qwen_engine(cfg.qwen.model_id, cfg.qwen.min_pixels, cfg.qwen.max_pixels,
                                 cfg.qwen.local_files_only)
    engine.cfg = cfg.qwen          # tham số runtime (số trường/lần gọi, thumbnail...) áp dụng ngay, không nạp lại
    return engine


STAGE_LABELS = [("render", "1 · Render PDF"), ("tesseract", "2 · Tesseract TSV"),
                ("qwen", "3 · Qwen2-VL local"), ("fusion", "4 · Fusion engine")]
STAGE_SPAN = {"render": (0.00, 0.05), "tesseract": (0.05, 0.35), "qwen": (0.35, 0.92), "fusion": (0.92, 1.00)}


def step_pipeline(cfg: AppConfig) -> None:
    st.subheader("Bước 3 · Thực thi pipeline")
    tpl: Optional[Template] = st.session_state.template
    if not tpl or not st.session_state.doc_bytes:
        st.info("Cần có template (Bước 1) và tài liệu (Bước 2).")
        return
    st.write(f"Tài liệu **{st.session_state.doc_name}** · template **{tpl.name}** · "
             f"Qwen: `{cfg.qwen.model_id}` (CPU float32, offline)")

    if st.button("🚀 Chạy trích xuất", type="primary"):
        bar = st.progress(0.0, text="Khởi động...")
        board = st.empty()
        done_stages: set = set()

        def render_board(active: str) -> None:
            lines = []
            for key, label in STAGE_LABELS:
                icon = "✅" if key in done_stages else ("⏳" if key == active else "⬜")
                lines.append(f"{icon} {label}")
            board.markdown("  \n".join(lines))

        def on_progress(stage: str, frac: float, msg: str) -> None:
            lo, hi = STAGE_SPAN[stage]
            bar.progress(min(1.0, lo + (hi - lo) * frac), text=msg)
            if frac >= 1.0:
                done_stages.add(stage)
            render_board(stage)

        try:
            engine = get_qwen_engine(cfg)
            with st.spinner("Đang nạp Qwen2-VL từ Hugging Face cache (lần đầu có thể mất 10-60 giây)..."):
                engine.load()
            pipe = OCRPipeline(cfg, engine)
            run = pipe.run(st.session_state.doc_bytes, st.session_state.doc_name, tpl, progress=on_progress)
            on_progress("fusion", 0.0, "Fusion: đồng thuận / trọng tài / confidence-weighted...")
            t0 = time.perf_counter()
            results = FusionEngine(cfg.fusion).fuse_document(tpl, run)
            run.timings["fusion_s"] = time.perf_counter() - t0
            on_progress("fusion", 1.0, "Hoàn tất")
            st.session_state.update(run=run, results=results, final=list(results), pipeline_error="",
                                    run_id=st.session_state.run_id + 1)
            bar.progress(1.0, text="Hoàn tất")
        except (PipelineError, TemplateError) as exc:
            st.session_state.pipeline_error = str(exc)
        except Exception as exc:  # lỗi ngoài dự kiến: hiển thị đầy đủ để debug khi demo
            st.session_state.pipeline_error = f"{type(exc).__name__}: {exc}"
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())

    if st.session_state.pipeline_error:
        st.error(st.session_state.pipeline_error)

    run: Optional[ExtractionRun] = st.session_state.run
    results: Optional[List[FieldResult]] = st.session_state.results
    if run and results:
        t = run.timings
        total = float(t.get("pipeline_s", 0)) + float(t.get("fusion_s", 0))
        cols = st.columns(6)
        cols[0].metric("Số trang", run.n_pages)
        cols[1].metric("Render", f"{t.get('render_s', 0):.1f}s")
        cols[2].metric("Tesseract", f"{t.get('tesseract_s', 0):.1f}s")
        cols[3].metric("Qwen2-VL", f"{t.get('qwen_s', 0):.1f}s", help=f"{t.get('qwen_calls', 0)} lần gọi")
        cols[4].metric("Fusion", f"{t.get('fusion_s', 0) * 1000:.0f}ms")
        cols[5].metric("Tổng", f"{total:.1f}s")
        counts = {s: sum(1 for r in results if r.validation_status == s) for s in ("ok", "warning", "error", "missing")}
        st.markdown(" ".join(f"{pill(k)} **{v}**" for k, v in counts.items()) +
                    f" &nbsp;·&nbsp; cần xem lại: **{sum(r.needs_review for r in results)}**",
                    unsafe_allow_html=True)
        for w in run.warnings:
            st.warning(w)
        st.button("Tiếp tục: review & sửa ▶", on_click=goto, args=(3,), type="primary")


# --------------------------------------------------------------------------- #
# Bước 4 – Review (human-in-the-loop)
# --------------------------------------------------------------------------- #
def _overlay(run: ExtractionRun, page: Optional[int], bbox) -> Optional[Image.Image]:
    if not page or not run.page_images or page > len(run.page_images):
        return None
    img = run.page_images[page - 1].copy()
    if bbox:
        d = ImageDraw.Draw(img)
        d.rectangle(tuple(bbox), outline=(220, 38, 38), width=max(4, img.width // 300))
    img.thumbnail((900, 1300))
    return img


def step_review() -> None:
    st.subheader("Bước 4 · Review & sửa lỗi (human-in-the-loop)")
    run: Optional[ExtractionRun] = st.session_state.run
    base: Optional[List[FieldResult]] = st.session_state.results
    tpl: Optional[Template] = st.session_state.template
    if not run or not base or not tpl:
        st.info("Hãy chạy pipeline ở Bước 3 trước.")
        return
    st.caption("Nhấp đúp ô **value** để sửa. Ô được sửa tự đổi `source = manual`, `confidence = 1.0` và được kiểm tra lại "
               "validator. Cột ⚠ đánh dấu trường nghi ngờ (thiếu / lỗi / confidence thấp / đã chọn bằng trọng tài).")
    only_review = st.checkbox("Chỉ hiện trường cần xem lại", value=False)

    rows = []
    for r in base:
        flag = "⚠" if (r.needs_review or r.validation_status != "ok") else ""
        rows.append({"⚠": flag, **r.to_row()})
    df = pd.DataFrame(rows)
    df["page"] = df["page"].astype(str)
    view = df[df["⚠"] == "⚠"] if only_review else df
    locked = [c for c in df.columns if c != "value"]
    edited = st.data_editor(
        view, hide_index=True, **STRETCH, disabled=locked, key=f"editor_{st.session_state.run_id}",
        column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0,
                                                                       format="%.2f"),
                       "value": st.column_config.TextColumn("value ✏️", width="large"),
                       "raw_tesseract": st.column_config.TextColumn(width="medium"),
                       "raw_qwen": st.column_config.TextColumn(width="medium")})

    # Áp các chỉnh sửa lên bản gốc (khoá theo field_key nên lọc/sắp xếp không làm lệch dòng)
    final: List[FieldResult] = []
    edited_by_key = {r["field_key"]: r["value"] for _, r in edited.iterrows()}
    n_manual = 0
    for r in base:
        spec = tpl.get(r.field_key)
        new_val = edited_by_key.get(r.field_key, r.value)
        if spec is not None and not spec.is_table and (new_val or "") != (r.value or ""):
            r = apply_manual_edit(r, spec, new_val)
            n_manual += 1
        final.append(r)

    # Bảng con
    for i, r in enumerate(final):
        if r.field_type != "table":
            continue
        with st.expander(f"📋 Bảng `{r.field_key}` ({len(r.table_rows or [])} hàng) – {r.validation_status}", expanded=False):
            cols = r.table_columns or []
            tdf = pd.DataFrame(r.table_rows or [], columns=cols)
            new_tdf = st.data_editor(tdf, num_rows="dynamic", **STRETCH, hide_index=True,
                                     key=f"tbl_{st.session_state.run_id}_{r.field_key}")
            if not new_tdf.fillna("").astype(str).equals(tdf.fillna("").astype(str)):
                spec = tpl.get(r.field_key)
                rows_new = new_tdf.fillna("").astype(str).to_dict("records")
                bad = 0
                for row in rows_new:
                    for c in (spec.columns if spec else []):
                        v = postprocess_value(c, row.get(c.column_key, "")).value
                        row[c.column_key] = v
                        bad += not validate_value(c.validation, v, c.type).passed
                final[i] = replace(r, table_rows=rows_new, source="manual", confidence=1.0,
                                   validation_status="ok" if not bad else "warning", needs_review=bool(bad),
                                   strategy="manual_edit", evidence="Cán bộ sửa bảng.",
                                   value=f"[bảng {len(rows_new)} hàng × {len(cols)} cột] -> sheet table_{r.field_key}")
                n_manual += 1
    st.session_state.final = final
    st.info(f"Số trường đã sửa tay: **{n_manual}**")

    # Bằng chứng fusion + overlay vùng trích
    st.markdown("##### 🔎 Bằng chứng chọn kết quả")
    key = st.selectbox("Chọn trường", [r.field_key for r in final], label_visibility="collapsed")
    sel = next(r for r in final if r.field_key == key)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**{sel.field_name}** · {pill(sel.validation_status)} · chiến lược `{sel.strategy}` · "
                    f"nguồn `{sel.source}` · confidence **{sel.confidence:.2f}**", unsafe_allow_html=True)
        st.write(sel.evidence or "—")
        st.json({"raw_tesseract": sel.raw_tesseract, "raw_qwen": sel.raw_qwen, "value": sel.value}, expanded=False)
    with c2:
        ov = _overlay(run, sel.page, sel.bbox)
        if ov is not None:
            st.image(ov, caption=f"Trang {sel.page}" + (" · khung đỏ = vùng Tesseract đọc" if sel.bbox else ""))
    st.button("Tiếp tục: xuất Excel ▶", on_click=goto, args=(4,), type="primary")


# --------------------------------------------------------------------------- #
# Bước 5 – Xuất
# --------------------------------------------------------------------------- #
def step_export() -> None:
    st.subheader("Bước 5 · Xuất Excel")
    run: Optional[ExtractionRun] = st.session_state.run
    final: Optional[List[FieldResult]] = st.session_state.final or st.session_state.results
    tpl: Optional[Template] = st.session_state.template
    if not run or not final or not tpl:
        st.info("Chưa có kết quả để xuất.")
        return
    try:
        data = export_to_bytes(final, run, tpl)
    except Exception as exc:
        st.error(f"Lỗi tạo file Excel: {exc}")
        st.code(traceback.format_exc())
        return
    stem = Path(st.session_state.doc_name).stem
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Tải file Excel (.xlsx)", data, file_name=f"{stem}_RESULT.xlsx", type="primary",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       on_click=lambda: append_trace_log(LOG_DIR / "trace.jsonl", final, run))
    c2.download_button("⬇️ Tải JSON (value, confidence, page, bbox)", results_to_json(final, run),
                       file_name=f"{stem}_RESULT.json", mime="application/json")
    st.markdown("**Nội dung file:** `Ket_qua` (9 cột chuẩn) · `table_<key>` (bảng) · `full_text` (văn bản thô 2 engine) · "
                "`metadata` (thời gian, cấu hình, phần cứng) · `fusion_audit` (bằng chứng lựa chọn).")
    preview = pd.DataFrame([r.to_row() for r in final])[OUTPUT_COLUMNS]
    preview["page"] = preview["page"].astype(str)
    st.dataframe(preview, hide_index=True, **STRETCH)


# --------------------------------------------------------------------------- #
def main() -> None:
    _init_state()
    cfg = sidebar()
    st.title("OCR tài liệu ngân hàng · Tesseract + Qwen2-VL")
    draw_stepper()
    idx = step_index()
    if idx == 0:
        step_template()
    elif idx == 1:
        step_document()
    elif idx == 2:
        step_pipeline(cfg)
    elif idx == 3:
        step_review()
    else:
        step_export()


main()
