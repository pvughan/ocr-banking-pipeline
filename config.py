"""
config.py
=========
Cấu hình tập trung cho toàn bộ pipeline OCR-PIPELINE-2026.

* Mọi tham số (DPI, PSM, kích thước ảnh đưa vào Qwen, ngưỡng fusion...) nằm ở đây
  để có thể ghi vào sheet `metadata` và tái lập kết quả (đề bài mục 7: "kết quả
  phải tái lập được từ code").
* Prompt của Qwen-VL được VERSION hoá ở đây (đề bài mục 5.4: "Prompt là một phần
  sản phẩm — phải version trong code").
* Có thể ghi đè bằng file JSON/YAML: `AppConfig.load("config.json")`.

Ràng buộc môi trường (macOS Apple Silicon, CPU-only):
    torch_dtype = float32, device_map = "cpu", KHÔNG dùng CUDA.
    Qwen2-VL: min_pixels = 224*224, max_pixels = 448*448.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PIPELINE_VERSION = "1.0.0"
PROMPT_VERSION = "qwen_extract_v1.0.0"

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"


# --------------------------------------------------------------------------- #
# Các nhóm cấu hình
# --------------------------------------------------------------------------- #
@dataclass
class RenderConfig:
    dpi: int = 200                      # đề bài: >= 200 DPI
    max_pages: int = 40                 # B1 = 16 trang, bundle = 34 trang
    poppler_path: Optional[str] = None  # None = tự dò (Homebrew: /opt/homebrew/bin)


@dataclass
class PreprocessConfig:
    enabled: bool = True
    auto_orient: bool = True            # sửa trang scan ngược 90/180/270 (Tesseract OSD)
    crop_border: bool = True            # cắt viền đen quanh trang scan
    deskew: bool = True                 # khử nghiêng (projection profile)
    suppress_red: bool = True           # dùng kênh R làm ảnh xám -> tem đỏ "biến mất"
    normalize_background: bool = True   # chia cho nền ước lượng -> khử bóng mép gáy
    denoise: bool = True                # median blur 3x3
    clahe: bool = True                  # tăng contrast cục bộ
    adaptive_threshold: bool = False    # nhị phân hoá thích nghi (thường chỉ cần cho bw150)
    target_dpi: int = 200               # ảnh có DPI thấp hơn sẽ được upscale
    max_upscale: float = 2.5
    max_deskew_deg: float = 5.0


@dataclass
class TesseractConfig:
    lang: str = "vie"
    fallback_lang: str = "eng"          # chỉ dùng nếu máy chưa cài 'vie' (sẽ có cảnh báo)
    oem: int = 1                        # LSTM
    psm_list: Tuple[int, ...] = (6,)    # thử nhiều PSM -> chọn PSM có conf trung bình cao nhất
    timeout_s: int = 180
    lookahead_lines: int = 2            # số dòng đọc thêm sau anchor
    anchor_min_score: float = 88.0      # ngưỡng fuzzy (0-100) khi dò anchor
    table_max_rows: int = 80


@dataclass
class QwenConfig:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    local_files_only: bool = True       # OFFLINE tuyệt đối: không tải gì từ Internet
    device: str = "cpu"
    dtype: str = "float32"
    min_pixels: int = 224 * 224
    max_pixels: int = 448 * 448
    thumb_max: int = 480                # thumbnail tạm trước khi đưa vào processor (400-500px)
    max_new_tokens: int = 320           # trần; số thực dùng tuỳ số trường trong chunk
    fields_per_call: int = 6            # chia nhỏ trường -> JSON ngắn -> ít ảo giác, nhanh hơn
    page_text_chars: int = 700          # trích Tesseract text của trang đưa vào prompt
    hint_chars: int = 160               # độ dài gợi ý OCR quanh anchor cho mỗi trường
    max_calls_per_doc: int = 40         # chặn trần số lần gọi Qwen / tài liệu (bảo vệ CPU)
    search_all_pages: bool = True       # trường chưa thấy anchor -> thử lần lượt các trang
    region_refine: bool = True          # đọc lại vùng cắt quanh anchor khi Qwen bỏ sót
    max_region_calls: int = 6
    region_pad_px: int = 24
    num_threads: int = 0                # 0 = dùng tất cả nhân CPU


@dataclass
class FusionConfig:
    fuzzy_threshold: float = 0.90       # khớp mờ >= 0.90 -> consensus
    exact_conf_range: Tuple[float, float] = (0.97, 1.00)
    fuzzy_conf_range: Tuple[float, float] = (0.95, 0.97)
    tess_trust_conf: float = 0.85       # Tesseract conf > 85% + qua validator -> chọn Tesseract
    review_threshold: float = 0.80      # confidence < ngưỡng -> needs_review
    single_source_cap: float = 0.75
    ungrounded_qwen_cap: float = 0.40   # Qwen đọc ra nhưng không có trong OCR text -> nghi ảo giác
    arbitration_margin_warning: float = 0.15
    keep_verbatim_value: bool = False   # True: value = nguyên văn (không chuẩn hoá) như file Lưu ý yêu cầu


@dataclass
class AppConfig:
    render: RenderConfig = field(default_factory=RenderConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        cfg = cls()
        for section in dc_fields(cfg):
            sub = data.get(section.name)
            if not isinstance(sub, dict):
                continue
            obj = getattr(cfg, section.name)
            valid = {f.name for f in dc_fields(obj)}
            for k, v in sub.items():
                if k not in valid:
                    continue
                current = getattr(obj, k)
                if isinstance(current, tuple) and isinstance(v, list):
                    v = tuple(v)
                setattr(obj, k, v)
        return cfg

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AppConfig":
        """Đọc cấu hình từ JSON/YAML. Không có file -> cấu hình mặc định."""
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Cần `pip install pyyaml` để đọc cấu hình YAML") from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)


# --------------------------------------------------------------------------- #
# Prompt Qwen-VL (VERSIONED) – KHÔNG viết prompt rải rác ở nơi khác
# --------------------------------------------------------------------------- #
PROMPTS: Dict[str, str] = {
    "version": PROMPT_VERSION,
    "system": (
        "Bạn là hệ thống OCR trích xuất thông tin có cấu trúc từ tài liệu ngân hàng và "
        "pháp lý bằng tiếng Việt. Bạn chỉ trả lời bằng MỘT đối tượng JSON hợp lệ."
    ),
    "page_instruction": (
        "Đọc ảnh trang tài liệu và trích xuất các trường dưới đây.\n"
        "QUY TẮC:\n"
        "1. Chỉ trả về JSON đúng schema, không giải thích, không markdown.\n"
        "2. Chép NGUYÊN VĂN giá trị như in/viết trên tài liệu (giữ dấu tiếng Việt, "
        "định dạng ngày và số tiền như hiển thị).\n"
        "3. Nếu không thấy trường trên trang này, trả về null. TUYỆT ĐỐI không đoán hay bịa số.\n"
        "4. Trường checkbox: trả về đúng một giá trị trong danh sách cho phép (hoặc null).\n"
        "5. Trường bảng: trả về mảng các đối tượng, mỗi hàng một đối tượng, đúng tên cột.\n\n"
        "DANH SÁCH TRƯỜNG:\n{field_guide}\n\n"
        "SCHEMA JSON CẦN TRẢ VỀ:\n{schema}\n"
        "{page_text_block}"
    ),
    "page_text_block": (
        "\nVĂN BẢN OCR THAM KHẢO CỦA TRANG (Tesseract, có thể sai chính tả; "
        "ưu tiên nội dung nhìn thấy trong ảnh):\n<<<\n{page_text}\n>>>\n"
    ),
    "region_instruction": (
        "Ảnh này là một vùng cắt từ tài liệu. Hãy đọc giá trị của trường \"{field_name}\" "
        "({type_hint}). Chỉ trả về JSON dạng {{\"value\": \"...\"}}, chép nguyên văn; "
        "nếu không đọc được trả về {{\"value\": null}}."
    ),
}
