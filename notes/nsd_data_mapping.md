# NSD Data Mapping & Context Notes

Tài liệu này tổng hợp logic mapping giữa hình ảnh (stimuli) và dữ liệu fMRI (time-series) trong bộ dữ liệu Natural Scenes Dataset (NSD), cùng các lưu ý quan trọng để tránh sai sót khi xử lý dữ liệu.

## 1. Cơ Chế Mapping (Mapping Logic)

Để kết nối giữa tín hiệu não tại một thời điểm và hình ảnh subject đang nhìn thấy, ta dựa vào file **Design Matrix** (`.tsv`) và file **Stimuli** (`.hdf5`).

*   **Logic**:
    *   Mỗi lượt chạy (run) có một file thiết kế (`.tsv`) chứa một cột các số nguyên.
    *   Mỗi dòng trong file `.tsv` tương ứng với một **Volume** (tiếng động) fMRI.
    *   **Giá trị `0`**: Khoảng nghỉ (Blank/Rest) hoặc không có sự kiện mới.
    *   **Giá trị `N > 0`**: Subject bắt đầu nhìn thấy hình ảnh có ID là `N`.
    *   **Lưu ý Indexing**: ID `N` trong file `.tsv` là **1-based index**.

## 2. Các Đường Dẫn Quan Trọng (Key Paths)

| Loại Dữ Liệu | Pattern Đường Dẫn | Ví dụ Cụ Thể (Subject 1) |
| :--- | :--- | :--- |
| **Logic Thiết Kế**<br>(Design Matrix) | `data/NSD/data/nsddata_timeseries/ppdata/subj{XX}/func1pt8mm/design/design_session{YY}_run{ZZ}.tsv` | `.../subj01/func1pt8mm/design/design_session01_run01.tsv` |
| **Dữ liệu fMRI**<br>(Time-series) | `data/NSD/data/nsddata_timeseries/ppdata/subj{XX}/func1pt8mm/timeseries/timeseries_session{YY}_run{ZZ}.nii.gz` | `.../subj01/func1pt8mm/timeseries/timeseries_session01_run01.nii.gz` |
| **Kho Ảnh Gốc**<br>(Stimuli Images) | `data/NSD/data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5` | `.../nsd_stimuli.hdf5` |

## 3. Các Lưu Ý Quan Trọng (Critical Caveats)

### 3.1. Sự chênh lệch kích thước (Padding)
*   File fMRI Timeseries thường có nhiều hơn file Design Matrix **1 volume** ở cuối.
    *   *Ví dụ*: Design có **225** dòng $\rightarrow$ Timeseries có **226** volumes.
*   **Nguyên nhân**: Do quy trình pre-processing.
*   **Xử lý**: Khi map, cần cắt bỏ hoặc phớt lờ volume cuối cùng của file timeseries để khớp dimension.

### 3.2. Phân biệt các loại thí nghiệm
Đừng nhầm lẫn giữa các file data nằm chung thư mục `timeseries`:
*   **NSD Core (`sessionXX`)**: Thí nghiệm chính (73k images). File bắt đầu bằng `timeseries_session...`.
*   **NSD Imagery (`nsdimagery`)**: Thí nghiệm tưởng tượng. File bắt đầu bằng `timeseries_nsdimagery...`. Số lượng volume thường là **240** hoặc **480**, khác với NSD Core.
*   **Functional Localizer (`prffloc`)**: Thí nghiệm định vị vùng não.
*   **NSD Synthetic (`nsdsynthetic`)**: Thí nghiệm ảnh tổng hợp.

### 3.3. Truy xuất ảnh từ HDF5
Khi lấy ảnh từ file `nsd_stimuli.hdf5` dựa trên ID từ file `.tsv`:
```python
# Giả sử id_from_tsv là giá trị đọc được (VD: 500)
image_index = id_from_tsv - 1 # Chuyển về 0-based index
image_data = hdf5_file['imgBrick'][image_index]
```

## 4. Các File Design Phụ Trợ
*   `design_floc_runCC.tsv`: Mapping cho thí nghiệm fLoc (Category 1-10).
*   `design_nsdsynthetic_runCC.tsv`: Mapping cho thí nghiệm ảnh tổng hợp (ID 1-284).
