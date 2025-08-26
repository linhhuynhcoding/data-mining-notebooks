import pandas as pd

# Đường dẫn đến file Parquet
file_path = 'duong_dan_den_file_cua_ban.parquet'

try:
    # Đọc file Parquet và lưu vào DataFrame của Pandas
    df = pd.read_parquet(file_path)

    # Hiển thị 5 dòng đầu tiên của DataFrame
    print(df.head())

    # In ra thông tin tổng quan về DataFrame
    print(df.info())

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn: {file_path}")
except Exception as e:
    print(f"Đã xảy ra lỗi khi đọc file: {e}")