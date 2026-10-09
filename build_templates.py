"""
build_templates.py – sinh template mẫu B1 và B5 (B2 dùng đúng file mẫu của đề bài).

    python build_templates.py [đường_dẫn_template_B2.xlsx]

Danh sách trường lấy từ file "Tổng hợp tài liệu và trường OCR" (sheet 'Trường cần bóc tách'): ưu tiên các
trường 'Bắt buộc đối chiếu = Có' + các trường dễ đo. Trường lặp (ben_the_chap[], danh_muc_ban_giao[]) được
mô hình hoá là type=table với sheet phụ `table_<key>`.

!! LƯU Ý QUAN TRỌNG: cột `anchor` và `page` ở đây là GỢI Ý dựa trên cách gọi tên thường gặp trong hồ sơ
   công chứng. Hãy mở PDF thật và chỉnh anchor cho khớp nhãn in sẵn (hoặc dùng trình soạn template trong
   app). Anchor tốt là yếu tố quyết định chất lượng của nhánh Tesseract.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from config import TEMPLATE_DIR
from template_parser import ColumnSpec, FieldSpec, Template, parse_template, template_to_excel_bytes

CCCD = "cccd12"
ROMAN = "Proper-case tiếng Việt"


def F(key, name, ftype="text", req="N", valid="", anchor="", page="all", notes="", cols=None) -> FieldSpec:
    return FieldSpec(key, name, ftype, req == "Y", valid, anchor, page, notes, cols or [])


def C(key, name, ctype="text", valid="", notes="") -> ColumnSpec:
    return ColumnSpec(key, name, ctype, valid, notes)


PARTY_COLS = [C("ho_ten", "Họ và tên", notes=ROMAN), C("ngay_sinh", "Ngày sinh", "date"),
              C("so_cccd", "Số CCCD", "text", CCCD), C("ngay_cap_cccd", "Ngày cấp CCCD", "date"),
              C("noi_thuong_tru", "Nơi thường trú")]

B1 = [
    F("so_hop_dong", "Số hợp đồng", "text", "Y", r"^\d{4}/[A-ZĐ]{2,6}/\d{3,6}$", "Số:", "1", "Ví dụ 2026/HĐBĐ/0456"),
    F("ngay_ky", "Ngày ký hợp đồng", "date", "Y", "", "ngày", "1", "dd/MM/yyyy"),
    F("dia_diem_ky", "Địa điểm ký", "text", "N", "", "tại", "1"),
    F("so_ban_hop_dong", "Số bản hợp đồng", "number", "N", "^\\d+$", "bản", "all"),
    F("ben_the_chap", "Danh sách bên thế chấp", "table", "Y", "", "BÊN THẾ CHẤP", "1-2",
      "columns=ho_ten|ngay_sinh|so_cccd|ngay_cap_cccd|noi_thuong_tru", PARTY_COLS),
    F("ben_nhan_ten", "Tên bên nhận thế chấp", "text", "N", "", "BÊN NHẬN THẾ CHẤP", "1-2", ROMAN),
    F("ben_nhan_dia_chi", "Địa chỉ bên nhận thế chấp", "text", "N", "", "Địa chỉ", "1-2"),
    F("ben_nhan_ma_so_thue", "Mã số thuế bên nhận", "text", "N", "msdn10", "Mã số thuế", "1-2"),
    F("ben_nhan_nguoi_dai_dien", "Người đại diện bên nhận", "text", "N", "", "Đại diện", "1-2", ROMAN),
    F("ben_nhan_chuc_vu", "Chức vụ người đại diện", "text", "N", "", "Chức vụ", "1-2"),
    F("dia_chi_tai_san", "Địa chỉ tài sản thế chấp", "text", "Y", "", "Địa chỉ", "2-4", "Giữ nguyên cấp quận của GCN"),
    F("thua_dat_so", "Thửa đất số", "number", "Y", "^\\d+$", "Thửa đất số", "2-4"),
    F("to_ban_do_so", "Tờ bản đồ số", "number", "Y", "^\\d+$", "Tờ bản đồ số", "2-4"),
    F("dien_tich_dat_m2", "Diện tích đất (m²)", "number", "Y", "number", "Diện tích", "2-4"),
    F("muc_dich_su_dung", "Mục đích sử dụng đất", "text", "N", "", "Mục đích sử dụng", "2-4"),
    F("thoi_han_su_dung", "Thời hạn sử dụng", "text", "N", "", "Thời hạn sử dụng", "2-4"),
    F("so_phat_hanh_gcn", "Số phát hành GCN", "text", "Y", r"^[A-Z]{1,3}\s?\d{5,8}$", "Số phát hành", "2-4"),
    F("so_vao_so_gcn", "Số vào sổ cấp GCN", "text", "Y", r"^[A-Z]{1,3}\s?\d{4,8}$", "Số vào sổ", "2-4"),
    F("ngay_cap_gcn", "Ngày cấp GCN", "date", "Y", "", "ngày cấp", "2-4"),
    F("co_quan_cap_gcn", "Cơ quan cấp GCN", "text", "Y", "", "Cơ quan cấp", "2-4"),
    F("loai_nha", "Loại tài sản gắn liền với đất", "text", "N", "", "Loại nhà", "2-4"),
    F("dien_tich_xay_dung", "Diện tích xây dựng", "number", "Y", "number", "Diện tích xây dựng", "2-4"),
    F("tong_dien_tich_san", "Tổng diện tích sàn", "number", "Y", "number", "Tổng diện tích sàn", "2-4"),
    F("gia_tri_dinh_gia_bang_so", "Tổng giá trị định giá (bằng số)", "money", "Y", "number", "Tổng giá trị", "3-6",
      "Số + VND; đối chiếu với bằng chữ"),
    F("gia_tri_dinh_gia_bang_chu", "Tổng giá trị định giá (bằng chữ)", "text", "Y", "", "Bằng chữ", "3-6"),
    F("gia_tri_quyen_su_dung_dat", "Giá trị quyền sử dụng đất", "money", "Y", "number", "quyền sử dụng đất", "3-6"),
    F("gia_tri_tai_san_gan_lien", "Giá trị tài sản gắn liền", "money", "Y", "number", "tài sản gắn liền", "3-6"),
    F("ngay_dinh_gia", "Ngày định giá", "date", "Y", "", "ngày định giá", "3-6"),
    F("don_vi_dinh_gia", "Đơn vị định giá", "text", "Y", "", "Đơn vị định giá", "3-6"),
    F("so_hop_dong_tin_dung", "Số hợp đồng tín dụng", "text", "Y", r"^\d{4}/[A-ZĐ]{2,6}/\d{3,6}$", "hợp đồng tín dụng", "3-8"),
    F("ngay_hop_dong_tin_dung", "Ngày hợp đồng tín dụng", "date", "Y", "", "ngày", "3-8"),
    F("so_tien_vay_bang_so", "Số tiền vay (bằng số)", "money", "Y", "number", "Số tiền vay", "3-8"),
    F("so_tien_vay_bang_chu", "Số tiền vay (bằng chữ)", "text", "Y", "", "Bằng chữ", "3-8"),
    F("muc_dich_vay", "Mục đích vay", "text", "N", "", "Mục đích vay", "3-8"),
    F("thoi_han_vay", "Thời hạn vay", "text", "N", "", "Thời hạn vay", "3-8"),
    F("to_chuc_cong_chung", "Tổ chức công chứng", "text", "N", "", "VĂN PHÒNG CÔNG CHỨNG", "all", ROMAN),
    F("cong_chung_vien", "Công chứng viên", "text", "N", "", "Công chứng viên", "all", ROMAN),
    F("so_cong_chung", "Số công chứng", "text", "Y", r"^\d{3,6}/\d{4}/[A-ZĐ]{2,6}$", "Số công chứng", "all"),
    F("quyen_so", "Quyển số", "text", "N", "", "Quyển số", "all"),
    F("ngay_cong_chung", "Ngày công chứng", "date", "Y", "", "Ngày công chứng", "all"),
    F("so_dieu_khoan", "Số điều khoản", "number", "N", "^\\d+$", "Điều", "all"),
    F("lich_tra_no_ky_dau_ngay", "Ngày trả kỳ đầu", "date", "Y", "", "Kỳ đầu", "13-14"),
    F("lich_tra_no_ky_cuoi_ngay", "Ngày trả kỳ cuối", "date", "Y", "", "Kỳ cuối", "13-14"),
    F("lich_tra_no_tong_goc", "Tổng gốc phải trả", "money", "N", "number", "Tổng", "13-14"),
]

B5 = [
    F("so_bien_ban", "Số biên bản", "text", "N", "", "Số:", "1"),
    F("ngay_lap", "Ngày lập biên bản", "date", "Y", "", "ngày", "1"),
    F("ben_giao", "Danh sách bên giao", "table", "Y", "", "BÊN GIAO", "1-2",
      "columns=ho_ten|ngay_sinh|so_cccd|ngay_cap_cccd|noi_thuong_tru", PARTY_COLS),
    F("ben_nhan_ten", "Tên bên nhận", "text", "N", "", "BÊN NHẬN", "1-2", ROMAN),
    F("ben_nhan_don_vi", "Đơn vị bên nhận", "text", "N", "", "Đơn vị", "1-2"),
    F("ben_nhan_nguoi_dai_dien", "Người đại diện bên nhận", "text", "N", "", "Đại diện", "1-2", ROMAN),
    F("ben_nhan_chuc_danh", "Chức danh bên nhận", "text", "N", "", "Chức danh", "1-2"),
    F("nguoi_chung_kien", "Người chứng kiến", "text", "N", "", "Người chứng kiến", "1-4", ROMAN),
    F("danh_muc_ban_giao", "Bảng danh mục bàn giao", "table", "N", "", "DANH MỤC", "1-3",
      "columns=stt|ten|loai_ban|so_luong",
      [C("stt", "STT", "number", "^\\d+$"), C("ten", "Tên hồ sơ/tài liệu"), C("loai_ban", "Loại bản"),
       C("so_luong", "Số lượng", "number", "^\\d+$")]),
    F("tong_so_loai_luu_giu", "Tổng số loại lưu giữ", "number", "N", "^\\d+$", "Tổng số loại", "2-4"),
    F("tong_so_ban_luu_giu", "Tổng số bản lưu giữ", "number", "N", "^\\d+$", "Tổng số bản", "2-4"),
    F("hop_dong_bao_hiem_so", "Số hợp đồng bảo hiểm", "text", "N", "", "hợp đồng bảo hiểm", "2-4"),
    F("hop_dong_bao_hiem_ngay", "Ngày hợp đồng bảo hiểm", "date", "Y", "", "ngày", "2-4"),
    F("nguoi_thu_huong", "Người thụ hưởng bảo hiểm", "text", "N", "", "Người thụ hưởng", "2-4", ROMAN),
    F("tong_chi_tieu", "Tổng chỉ tiêu đối chiếu", "number", "N", "^\\d+$", "Tổng chỉ tiêu", "3-4"),
    F("so_chi_tieu_khop", "Số chỉ tiêu khớp", "number", "N", "^\\d+$", "chỉ tiêu khớp", "3-4"),
    F("ket_luan", "Kết luận", "text", "N", "", "Kết luận", "3-4"),
    F("ma_ho_so", "Mã hồ sơ nhập kho", "text", "N", r"^[A-Z]{4}-\d{4}-\d{6}$", "Mã hồ sơ", "3-4"),
    F("ngay_nhap_kho", "Ngày nhập kho", "date", "Y", "", "Ngày nhập kho", "3-4"),
    F("gio_nhap_kho", "Giờ nhập kho", "text", "N", "", "Giờ nhập kho", "3-4"),
    F("vi_tri_luu_tru", "Vị trí lưu trữ", "text", "N", "", "Vị trí lưu trữ", "3-4"),
    F("can_bo_nhap_kho", "Cán bộ nhập kho", "text", "N", "", "Cán bộ nhập kho", "3-4", ROMAN),
    F("kiem_soat_vien", "Kiểm soát viên", "text", "N", "", "Kiểm soát viên", "3-4", ROMAN),
]


def build(name: str, fields: list) -> Path:
    tpl = Template(name=name, fields=fields)
    out = TEMPLATE_DIR / f"{name}.xlsx"
    out.write_bytes(template_to_excel_bytes(tpl))
    parsed = parse_template(out)          # kiểm tra vòng khứ hồi
    assert len(parsed.fields) == len(fields), "round-trip template lỗi"
    return out


def main() -> None:
    TEMPLATE_DIR.mkdir(exist_ok=True)
    for name, fields in (("B1_HopDongTheChap", B1), ("B5_BienBanGiaoNhan", B5)):
        print("Đã tạo", build(name, fields), f"({len(fields)} trường)")
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        dst = TEMPLATE_DIR / "B2_PhieuYeuCau.xlsx"
        shutil.copy(sys.argv[1], dst)
        print("Đã chép template B2 ->", dst)


if __name__ == "__main__":
    main()
