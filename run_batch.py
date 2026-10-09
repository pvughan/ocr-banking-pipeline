"""
run_batch.py – chạy pipeline hàng loạt (không cần UI) để sinh file Excel kết quả cho bộ 10 file test.

    python run_batch.py --input DataTest_AI_OCR/ --templates templates/ --out outputs/
    python run_batch.py --input B2_scan.pdf --template templates/B2_PhieuYeuCau.xlsx --out outputs/

* Template được chọn theo mã ở đầu tên file (B1_scan.pdf -> templates/B1_*.xlsx) hoặc `--template`.
* Mỗi file -> `<tên_file>.xlsx` (đủ sheet Ket_qua/full_text/metadata/...) + `<tên_file>.json`.
* Model Qwen chỉ nạp MỘT lần cho cả lô. Nhật ký truy vết: logs/trace.jsonl.
Sau đó đánh giá:  python eval_metrics.py --gt GT.xlsx --pred outputs/ --templates templates/
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

from config import LOG_DIR, AppConfig
from excel_exporter import append_trace_log, export_to_file, results_to_json
from fusion import FusionEngine
from pipeline_engine import OCRPipeline, PipelineError, QwenVLEngine
from template_parser import Template, TemplateError, parse_template

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def pick_template(path: Path, templates_dir: Optional[Path], single: Optional[Path]) -> Template:
    if single:
        return parse_template(single)
    if templates_dir:
        m = re.match(r"(?i)^(B\d+)", path.stem)
        if m:
            for cand in sorted(templates_dir.glob(f"{m.group(1)}*.xlsx")):
                return parse_template(cand)
    raise TemplateError(f"Không chọn được template cho '{path.name}' (đặt tên bắt đầu bằng B1..B6 hoặc dùng --template)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Chạy OCR Tesseract + Qwen2-VL hàng loạt")
    ap.add_argument("--input", required=True, help="file hoặc thư mục chứa PDF/ảnh")
    ap.add_argument("--templates", help="thư mục template (chọn theo mã B1..B6 ở đầu tên file)")
    ap.add_argument("--template", help="một file template dùng cho tất cả")
    ap.add_argument("--out", default="outputs", help="thư mục ghi kết quả")
    ap.add_argument("--config", help="file cấu hình JSON/YAML (ghi đè mặc định)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = AppConfig.load(args.config)
    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"Không có PDF/ảnh trong {src}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = QwenVLEngine(cfg.qwen)
    pipe = OCRPipeline(cfg, engine)
    fusion = FusionEngine(cfg.fusion)
    failures = 0
    for i, f in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        try:
            tpl = pick_template(f, Path(args.templates) if args.templates else None,
                                Path(args.template) if args.template else None)
            run = pipe.run(f.read_bytes(), f.name, tpl,
                           progress=lambda s, fr, m: print(f"   {s:10} {fr:4.0%}  {m[:80]}", end="\r"),
                           keep_page_images=False)
            t0 = time.perf_counter()
            results = fusion.fuse_document(tpl, run)
            run.timings["fusion_s"] = time.perf_counter() - t0
            export_to_file(out_dir / f"{f.stem}.xlsx", results, run, tpl)
            (out_dir / f"{f.stem}.json").write_text(results_to_json(results, run), encoding="utf-8")
            append_trace_log(LOG_DIR / "trace.jsonl", results, run)
            st = {s: sum(1 for r in results if r.validation_status == s) for s in ("ok", "warning", "error", "missing")}
            print(f"\n   -> {f.stem}.xlsx · {st} · {run.timings['pipeline_s'] + run.timings['fusion_s']:.1f}s")
        except (PipelineError, TemplateError) as exc:
            failures += 1
            print(f"\n   LỖI: {exc}", file=sys.stderr)
        except Exception:
            failures += 1
            print("\n   LỖI không lường trước:", file=sys.stderr)
            traceback.print_exc()
    print(f"\nHoàn tất: {len(files) - failures}/{len(files)} file thành công -> {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
