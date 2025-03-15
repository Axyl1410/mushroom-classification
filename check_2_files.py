import pandas as pd
def compare_csv(original_file, predicted_file):
    # Đọc dữ liệu từ file CSV
    original_df = pd.read_csv(original_file)
    predicted_df = pd.read_csv(predicted_file)

    # Kiểm tra cột có khớp nhau không
    if original_df.columns.tolist() != predicted_df.columns.tolist():
        raise ValueError("Tên cột giữa hai file không khớp nhau.")

    # Kiểm tra số lượng dòng có khớp nhau không
    if len(original_df) != len(predicted_df):
        raise ValueError("Số dòng giữa hai file không khớp nhau.")

    # So sánh từng dòng
    matches = (original_df == predicted_df).all(axis=1).sum()

    # Tính toán độ chính xác
    accuracy = (matches / len(original_df)) * 100

    print(f"Số dòng khớp: {matches}/{len(original_df)}")
    print(f"Độ chính xác: {accuracy:.2f}%")

# Gọi hàm so sánh
compare_csv('mushroom_labels_trick.csv', 'mushroom_predictions.csv')
