"""
cli.py
======
Giao diện dòng lệnh (CLI) chạy trích xuất tài liệu ngân hàng bằng Dual-Engine OCR.

Cách dùng:
    python cli.py --input document.pdf --output result.xlsx
    python cli.py --input document.pdf --template templates/Template_B2_PhieuYeuCau.xlsx --output result.xlsx
    python cli.py --input folder_pdf/ --output results/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import AppConfig, PIPELINE_VERSION
from excel_exporter import export_results
from fusion import FusionEngine
from pipeline_engine import OCRPipeline
from template_parser import parse_template


def main():
    parser = argparse.ArgumentParser(description=f"AI Banking OCR Pipeline CLI (v{PIPELINE_VERSION})")
    parser.add_argument("--input", "-i", required=True, help="Đường dẫn file PDF / ảnh hoặc thư mục chứa nhiều file")
    parser.add_argument("--output", "-o", required=True, help="Đường dẫn file Excel đầu ra (.xlsx) hoặc thư mục kết quả")
    parser.add_argument("--template", "-t", default=None, help="Đường dẫn file template Excel (.xlsx)")
    args = parser.parse_args()

    # 1. Nạp Template
    tpl_path = args.template
    if not tpl_path:
        default_tpl = Path(__file__).resolve().parent / "templates" / "Template_B2_PhieuYeuCau.xlsx"
        if default_tpl.exists():
            tpl_path = str(default_tpl)
        else:
            print("❌ Lỗi: Vui lòng chỉ định --template hoặc tạo template mẫu trong thư mục templates/")
            sys.exit(1)

    print(f"📋 Đang nạp template: {tpl_path}")
    template = parse_template(tpl_path)
    print(f"✅ Template: {template.name} ({len(template.fields)} trường)")

    # 2. Khởi tạo Pipeline & Fusion Engine
    cfg = AppConfig()
    pipeline = OCRPipeline(cfg)
    fusion = FusionEngine(cfg.fusion)

    # 3. Xác định danh sách file
    inp = Path(args.input)
    supported_exts = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}

    if inp.is_file():
        files = [inp]
        out_dir = None
    elif inp.is_dir():
        files = sorted([f for f in inp.iterdir() if f.suffix.lower() in supported_exts])
        if not files:
            print(f"❌ Không tìm thấy file tài liệu hợp lệ trong {inp}")
            sys.exit(1)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"❌ Không tìm thấy đường dẫn đầu vào: {args.input}")
        sys.exit(1)

    print(f"\n🚀 Bắt đầu xử lý {len(files)} tài liệu...")

    for idx, f in enumerate(files, start=1):
        print(f"\n--- [{idx}/{len(files)}] Xử lý: {f.name} ---")
        t0 = time.time()

        with open(f, "rb") as fp:
            data = fp.read()

        # Chạy pipeline
        run = pipeline.run(data=data, filename=f.name, template=template)

        # Chạy fusion
        results = fusion.fuse_document(template, run)

        # Xuất file kết quả Excel chuẩn 3 sheet
        if out_dir:
            out_file = out_dir / f"{f.stem}_result.xlsx"
        else:
            out_file = Path(args.output)

        export_results(results, run, template, out_file)
        elapsed = time.time() - t0

        print(f"✅ Hoàn tất trong {elapsed:.2f}s | Lưu file: {out_file}")
        print(f"   • Tesseract: {run.timings.get('tesseract_s', 0):.2f}s")
        print(f"   • Qwen2-VL : {run.timings.get('qwen_s', 0):.2f}s")
        print(f"   • Số trường trích xuất: {len(results)}")

    print(f"\n🎉 Hoàn thành toàn bộ {len(files)} tài liệu!")


if __name__ == "__main__":
    main()
