#Ứng dụng dự báo chứng khoán bằng Al
# Stock Price Forecasting with LSTM
Ứng dụng dự báo giá cổ phiếu bằng mô hình LSTM và Python.
## Công nghệ
- Python
- TensorFlow / Keras
- Streamlit
- Yahoo Finance
- Scikit-learn
- Pandas, NumPy, Matplotlib
## Chạy ứng dụng
Cài đặt thư viện:
```bash
pip install -r requirements.txt
```
Khởi chạy:
```bash
streamlit run app.py
```
## Tính năng
- Tải dữ liệu giá cổ phiếu từ Yahoo Finance
- Huấn luyện mô hình LSTM
- Đánh giá bằng RMSE, MAE và R²
- Hiển thị biểu đồ giá thực tế và giá dự báo
- Dự báo các phiên tiếp theo
- Xuất kết quả dưới dạng CSV

## Disclaimer
Kết quả dự báo chỉ mang tính tham khảo, không phải lời khuyên đầu tư.
