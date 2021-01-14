# author: quanvo
# year: 2020
# note:
### 0_Data (Dữ liệu)
- 1600 dữ liệu ý kiến đánh giá giảng viên của trường Đại học Công Nghệ Tp.HCM
- 500 dữ liệu ý kiến đánh giá tích cực
- 500 dữ liệu ý kiến đánh giá tiêu cực
- 821 văn bản ở nhiều chủ đề dùng để tạo model huấn luyện

### 1_Word_Segementation (Tiền xử lý)
    - Parse xml wiki data
    - Sửa lỗi font tiếng Việt
    - Chuẩn hóa dấu câu
    - Tokenize (Sử dụng ViTokenizer)
    - Loại bỏ các common token: email, phone, emoij, number,...

### 2_Classification
  - Biễu diễn các dữ liệu thành các mô hình:
    + Mô hình túi từ (bags of word)
    + Mô hình không gian vector (sentence2vec)
  - Chạy thực nghiệm với các thuật toán phân lớp: naive , svm, tree


# Mọi thắc mắc vui lòng liên hệ quanvo.dev@gmail.com
