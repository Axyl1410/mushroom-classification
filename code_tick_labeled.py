import os
import pandas as pd

# Đường dẫn thư mục chứa ảnh
image_root = r"D:\Testing\aio-hutech\train"

# Định nghĩa mapping thư mục -> nhãn
label_mapping = {
    'bao ngu xam trang': 1,
    'dui ga baby': 2,
    'linh chi trang': 3,
    'nam mo': 0
}

# Danh sách lưu thông tin file và nhãn
data = []

# Duyệt qua các thư mục và gán nhãn
for folder, label in label_mapping.items():
    folder_path = os.path.join(image_root, folder)

    if not os.path.exists(folder_path):
        print(f"Thư mục không tồn tại: {folder_path}")
        continue

    # Duyệt qua từng file trong thư mục
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Loại bỏ phần đuôi mở rộng và giữ nguyên tên
            name_without_ext = os.path.splitext(filename)[0]
            data.append([name_without_ext, label])

# Tạo DataFrame và xuất ra file CSV
labels_df = pd.DataFrame(data, columns=['id', 'type'])
labels_df.to_csv('mushroom_labels.csv', index=False)

print("Hoàn thành! Đã tạo file mushroom_labels.csv")
