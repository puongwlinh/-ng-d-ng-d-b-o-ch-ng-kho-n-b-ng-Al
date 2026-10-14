import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. CẤU HÌNH
# =========================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.set_page_config(
    page_title="Dự báo giá cổ phiếu",
    page_icon="📈",
    layout="wide"
)

st.title("📈 DỰ BÁO GIÁ CỔ PHIẾU BẰNG LSTM")
st.write(
    "Ứng dụng sử dụng dữ liệu lịch sử từ Yahoo Finance "
    "và mô hình học sâu LSTM để dự báo giá đóng cửa."
)

# =========================
# 2. GIAO DIỆN NHẬP DỮ LIỆU
# =========================
with st.sidebar:
    st.header("Cài đặt mô hình")

    ticker = st.text_input(
        "Mã cổ phiếu",
        value="VNM.VN"
    ).strip().upper()

    start_date = st.date_input(
        "Ngày bắt đầu",
        value=pd.Timestamp("2018-01-01").date()
    )

    end_date = st.date_input(
        "Ngày kết thúc",
        value=pd.Timestamp.today().date()
    )

    window_size = st.slider(
        "Số phiên lịch sử đầu vào",
        min_value=20,
        max_value=120,
        value=60
    )

    forecast_days = st.slider(
        "Số phiên muốn dự báo",
        min_value=1,
        max_value=30,
        value=10
    )

    epochs = st.slider(
        "Số vòng huấn luyện",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    run = st.button(
        "Bắt đầu dự báo",
        type="primary",
        use_container_width=True
    )


# =========================
# 3. HÀM TẠO CHUỖI DỮ LIỆU
# =========================
def create_sequences(data, window_size):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# =========================
# 4. TẢI VÀ XỬ LÝ DỮ LIỆU
# =========================
if run:

    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    with st.spinner("Đang tải dữ liệu và huấn luyện LSTM..."):

        try:
            df = yf.download(
                ticker,
                start=str(start_date),
                end=str(
                    pd.Timestamp(end_date)
                    + pd.Timedelta(days=1)
                ),
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                st.error(
                    "Không tìm thấy dữ liệu. "
                    "Hãy kiểm tra mã cổ phiếu và ngày đã chọn."
                )
                st.stop()

            close = df["Close"]

            # Xử lý dữ liệu MultiIndex của yfinance
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            prices = pd.DataFrame({
                "Close": pd.to_numeric(
                    close, errors="coerce"
                )
            }).dropna()

            data = prices[["Close"]].values.astype(
                np.float32
            )

            if len(data) <= window_size + 20:
                st.error(
                    "Dữ liệu không đủ. "
                    "Hãy chọn khoảng thời gian dài hơn "
                    "hoặc giảm số phiên đầu vào."
                )
                st.stop()

            # =========================
            # 5. CHIA TẬP TRAIN / TEST
            # =========================
            train_size = int(len(data) * 0.8)

            scaler = MinMaxScaler()

            train_scaled = scaler.fit_transform(
                data[:train_size]
            )

            all_scaled = scaler.transform(data)

            X_train, y_train = create_sequences(
                train_scaled, window_size
            )

            # Tạo tập test có đủ dữ liệu lịch sử
            test_combined = all_scaled[
                train_size - window_size:
            ]

            X_test, y_test = create_sequences(
                test_combined, window_size
            )

            # Tách validation theo thứ tự thời gian
            val_size = max(
                1, int(len(X_train) * 0.1)
            )

            if len(X_train) <= val_size:
                st.error(
                    "Không đủ dữ liệu để huấn luyện."
                )
                st.stop()

            X_train_final = X_train[:-val_size]
            y_train_final = y_train[:-val_size]

            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]

            # =========================
            # 6. XÂY DỰNG MÔ HÌNH LSTM
            # =========================
            model = Sequential([
                Input(shape=(window_size, 1)),
                LSTM(50, return_sequences=True),
                LSTM(50),
                Dense(1)
            ])

            model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train_final,
                y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            # =========================
            # 7. ĐÁNH GIÁ MÔ HÌNH
            # =========================
            test_pred_scaled = model.predict(
                X_test, verbose=0
            )

            test_pred = scaler.inverse_transform(
                test_pred_scaled
            ).ravel()

            y_test_real = scaler.inverse_transform(
                y_test
            ).ravel()

            rmse = np.sqrt(
                mean_squared_error(
                    y_test_real, test_pred
                )
            )

            mae = mean_absolute_error(
                y_test_real, test_pred
            )

            r2 = r2_score(
                y_test_real, test_pred
            )

            # =========================
            # 8. DỰ BÁO TƯƠNG LAI
            # =========================
            last_window = all_scaled[
                -window_size:
            ].reshape(1, window_size, 1)

            future_scaled = []

            for _ in range(forecast_days):

                next_pred = model.predict(
                    last_window, verbose=0
                )

                future_scaled.append(
                    next_pred[0, 0]
                )

                last_window = np.concatenate(
                    [
                        last_window[:, 1:, :],
                        next_pred.reshape(1, 1, 1)
                    ],
                    axis=1
                )

            future_prices = scaler.inverse_transform(
                np.array(
                    future_scaled
                ).reshape(-1, 1)
            ).ravel()

            future_dates = pd.bdate_range(
                start=prices.index[-1]
                + pd.Timedelta(days=1),
                periods=forecast_days
            )

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
            st.stop()

    # =========================
    # 9. HIỂN THỊ KẾT QUẢ
    # =========================
    st.success("Huấn luyện mô hình hoàn tất!")

    latest_price = float(prices["Close"].iloc[-1])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Giá đóng cửa gần nhất",
        f"{latest_price:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

    col3.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    col4.metric(
        "R²",
        f"{r2:.4f}"
    )

    # Biểu đồ giá lịch sử
    st.subheader("1. Giá cổ phiếu lịch sử")

    st.line_chart(
        prices["Close"],
        use_container_width=True
    )

    # Biểu đồ so sánh
    st.subheader(
        "2. Giá thực tế và giá dự báo trên tập kiểm tra"
    )

    test_dates = prices.index[train_size:]

    comparison = pd.DataFrame(
        {
            "Giá thực tế": y_test_real,
            "Giá dự báo": test_pred
        },
        index=test_dates
    )

    st.line_chart(
        comparison,
        use_container_width=True
    )

    # Biểu đồ loss
    st.subheader("3. Quá trình huấn luyện")

    loss_df = pd.DataFrame({
        "Train Loss": history.history["loss"],
        "Validation Loss": history.history[
            "val_loss"
        ]
    })

    st.line_chart(
        loss_df,
        use_container_width=True
    )

    # Bảng dự báo tương lai
    st.subheader(
        f"4. Dự báo {forecast_days} phiên tiếp theo"
    )

    forecast_df = pd.DataFrame({
        "Ngày dự kiến": future_dates.strftime(
            "%Y-%m-%d"
        ),
        "Giá dự báo": future_prices
    })

    st.dataframe(
        forecast_df.style.format({
            "Giá dự báo": "{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Biểu đồ tương lai
    st.subheader("5. Biểu đồ dự báo tương lai")

    fig, ax = plt.subplots(figsize=(12, 5))

    recent_n = min(120, len(prices))

    ax.plot(
        prices.index[-recent_n:],
        prices["Close"].iloc[-recent_n:],
        label="Giá lịch sử"
    )

    ax.plot(
        future_dates,
        future_prices,
        marker="o",
        linestyle="--",
        label="Giá dự báo"
    )

    ax.set_title(
        f"Dự báo giá cổ phiếu {ticker} bằng LSTM"
    )

    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá đóng cửa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Tải dữ liệu xuống
    csv = forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Tải kết quả dự báo CSV",
        data=csv,
        file_name=f"{ticker}_forecast.csv",
        mime="text/csv"
    )

    st.warning(
        "Lưu ý: Đây là kết quả dự báo từ dữ liệu lịch sử, "
        "không đảm bảo giá thực tế sẽ diễn biến tương tự "
        "và không phải lời khuyên đầu tư."
    )

else:
    st.markdown("""
    ### Hướng dẫn sử dụng

    1. Nhập mã cổ phiếu.
    2. Chọn khoảng thời gian lấy dữ liệu.
    3. Chọn số phiên lịch sử dùng làm đầu vào.
    4. Chọn số phiên tương lai cần dự báo.
    5. Nhấn **Bắt đầu dự báo** để xem kết quả.
    """)
