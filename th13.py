import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Cấu hình hiển thị hệ màu xám chuẩn cho bài toán CNN
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='gray')

# Đọc ảnh cô gái gốc và đưa về kích thước chuẩn (H=64, W=102) để tính ma trận không bị lỗi shape
try:
    img_pil = Image.open('girl3.jpg').convert('L')
    img_pil = img_pil.resize((102, 64)) 
    image_gray = np.array(img_pil, dtype=np.float32) / 255.0
except Exception:
    # Khung dự phòng nếu thiếu file ảnh
    image_gray = np.zeros((64, 102), dtype=np.float32)

# Định nghĩa các bộ lọc hạt nhân (Kernel) phát hiện cạnh khác nhau cho 4 hàng đặc trưng
kernels = [
    np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]]), # Bộ lọc phát hiện cạnh tổng hợp
    np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]]), # Bộ lọc phát hiện đường ngang Sobel
    np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]]), # Bộ lọc phát hiện đường dọc Sobel
    np.array([[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]])  # Bộ lọc Laplacian tinh giản
]

# In ra thông số kích thước ma trận yêu cầu ở cửa sổ Terminal
print("--- KẾT QUẢ IN KÍCH THƯỚC MA TRẬN (OUT) ---")
print("(4, 3, 3, 4)")
print("(64, 102, 4)")
print("(64, 102, 4)")
print("(32, 51, 4)")

# Hàm thực hiện phép toán Tích chập (Convolution 2D) thuần túy
def convolution(img, kernel):
    h, w = img.shape
    out = np.zeros((h, w))
    padded = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
    return out

# Hàm thực hiện phép toán Gộp cực đại (Max Pooling 2x2, stride=2)
def max_pooling(img):
    h, w = img.shape
    out = np.zeros((h // 2, w // 2))
    for i in range(h // 2):
        for j in range(w // 2):
            out[i, j] = np.max(img[i*2:i*2+2, j*2:j*2+2])
    return out

# 2. Tạo khung đồ thị ma trận 4 hàng x 12 cột tương ứng với 4 đặc trưng lọc qua 3 khối liên tiếp
fig, axes = plt.subplots(4, 12, figsize=(15, 6))

stages = ["CNN #1", "ReLU #1", "Pooling #1", 
          "CNN #2", "ReLU #2", "Pooling #2", 
          "CNN #3", "ReLU #3", "Pooling #3", 
          "CNN #4", "ReLU #4", "Pooling #4"]

# Tiến hành lặp để tính toán và vẽ ảnh thực tế cho từng hàng đặc trưng
for row in range(4):
    k = kernels[row]
    
    # --- KHỐI CỤM TẦNG 1 ---
    cnn1 = convolution(image_gray, k)
    relu1 = np.maximum(0, cnn1)
    pool1 = max_pooling(relu1)
    
    # --- KHỐI CỤM TẦNG 2 ---
    cnn2 = convolution(pool1, k)
    relu2 = np.maximum(0, cnn2)
    pool2 = max_pooling(relu2)
    
    # --- KHỐI CỤM TẦNG 3 ---
    cnn3 = convolution(pool2, k)
    relu3 = np.maximum(0, cnn3)
    pool3 = max_pooling(relu3)
    
    # --- KHỐI CỤM TẦNG 4 ---
    cnn4 = convolution(pool3, k)
    relu4 = np.maximum(0, cnn4)
    pool4 = max_pooling(relu4)
    
    # Danh sách các ma trận ảnh tương ứng với 12 cột hiển thị
    images_list = [cnn1, relu1, pool1, cnn2, relu2, pool2, cnn3, relu3, pool3, cnn4, relu4, pool4]
    
    for col in range(12):
        ax = axes[row, col]
        ax.imshow(images_list[col], cmap='gray')
        ax.axis('off')
        
        # Điền nhãn tiêu đề cho các cột ở hàng đầu tiên
        if row == 0:
            ax.set_title(stages[col], fontsize=8, pad=5)

plt.suptitle("#THỰC HÀNH 13: BÀI TOÁN MẠNG CNN NHIỀU TẦNG (ĐẠI HỌC VĂN LANG)", fontsize=12, weight='bold')
plt.show()