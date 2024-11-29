ML_SGU_project
==============================

Project Organization
------------

    ├── LICENSE
    ├── README.md              <- Giới thiệu tổng quan về dự án, cung cấp hướng dẫn cho các nhà phát triển hoặc người dùng muốn sử dụng dự án này.
    ├── data
    │   ├── processed          <- Bộ dữ liệu đã được xử lý
    │   ├── raw                <- Bộ dữ liệu thô
    │   └── external           <- Các dữ liệu khác     
    │
    ├── models                 <- Thư mục chứa ô hình đã được huấn luyện
    │
    │
    ├── notebooks
    │   ├── exploratory        <- Notebook EDA
    │   └── results            <- Trực quan hóa kết quả 
    │
    ├── reports                <- Thư mục chứa các báo cáo dạng HTML, PDF, LaTeX,...
    │   └── figures            <-  Chứa các hình ảnh sử dụng trong các báo cáo.
    │
    ├── requirements.txt       <- Tập tin chứa danh sách các gói Python cần thiết để tái tạo môi trường phân tích của dự án
    │
    ├── src                    <- Source code của dự án.
    │    ├── __init__.py       <- File giúp biến thư mục thành một module Python
    │    ├── config.py         <- Lưu trưc biến, tham số mô hình
    │    ├── data              <- Tải xuống hoặc tạo dữ liệu
    │    │
    │    ├── data_engineering 
    │    │
    │    └──  models            <- Thư mục con chứa mã nguồn liên quan đến mô hình
    │         ├── predict_model
    │         └── train_model
    │                └── logs   <- Nhật ký thí nghiệm
    │
    ├── api/                    <- Thư mục chứa mã nguồn cho API  
    
