import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ===== 1. Baca gambar =====
img = cv2.imread('patung2.jpg')
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
height, width, channels = img.shape
print(f"Ukuran gambar: {width}x{height}")

# Konversi ke RGB
print("Konversi ke RGB...")
img_rgb = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        B, G, R = img[i,j]
        img_rgb[i,j] = [R, G, B]
print("RGB selesai")

# Konversi ke grayscale asli
print("Konversi ke grayscale...")
gray = np.zeros((height, width), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        R, G, B = img_rgb[i,j]
        gray[i,j] = int(0.299*R + 0.587*G + 0.114*B)
print("Grayscale asli selesai\n")

# ===== Noise manual =====
def add_salt(img_rgb, amount):
    out = np.zeros_like(img_rgb)
    for i in range(height):
        for j in range(width):
            for c in range(3):
                out[i,j,c] = img_rgb[i,j,c]
            if np.random.rand() < amount:
                out[i,j,:] = 255
    return out

def add_pepper(img_rgb, amount):
    out = np.zeros_like(img_rgb)
    for i in range(height):
        for j in range(width):
            for c in range(3):
                out[i,j,c] = img_rgb[i,j,c]
            if np.random.rand() < amount:
                out[i,j,:] = 0
    return out

def add_gaussian(img_rgb, sigma):
    out = np.zeros_like(img_rgb)
    for i in range(height):
        for j in range(width):
            for c in range(3):
                noise = np.random.normal(0, sigma)
                val = img_rgb[i,j,c] + noise
                val = max(0, min(255, int(val)))
                out[i,j,c] = val
    return out

# Konversi RGB -> grayscale manual
def rgb2gray_manual(img_rgb):
    gray_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            R, G, B = img_rgb[i,j]
            gray_img[i,j] = int(0.299*R + 0.587*G + 0.114*B)
    return gray_img

# ===== Filter manual 5x5 =====
def min_filter_manual(img):
    out = np.zeros_like(img)
    padded = np.zeros((height+4, width+4), dtype=np.uint8)
    padded[2:-2,2:-2] = img
    for i in range(height):
        for j in range(width):
            kernel = padded[i:i+5,j:j+5]
            out[i,j] = np.min(kernel)
    return out

def max_filter_manual(img):
    out = np.zeros_like(img)
    padded = np.zeros((height+4, width+4), dtype=np.uint8)
    padded[2:-2,2:-2] = img
    for i in range(height):
        for j in range(width):
            kernel = padded[i:i+5,j:j+5]
            out[i,j] = np.max(kernel)
    return out

def mean_filter_manual(img):
    out = np.zeros_like(img)
    padded = np.zeros((height+4, width+4), dtype=np.uint8)
    padded[2:-2,2:-2] = img
    for i in range(height):
        for j in range(width):
            kernel = padded[i:i+5,j:j+5]
            out[i,j] = int(np.sum(kernel)/25)
    return out

def median_filter_manual(img):
    out = np.zeros_like(img)
    padded = np.zeros((height+4, width+4), dtype=np.uint8)
    padded[2:-2,2:-2] = img
    for i in range(height):
        for j in range(width):
            kernel = padded[i:i+5,j:j+5].flatten()
            kernel.sort()
            out[i,j] = kernel[len(kernel)//2]
    return out

# MSE manual
def mse_manual(original, filtered):
    total = 0
    for i in range(height):
        for j in range(width):
            total += (int(original[i,j]) - int(filtered[i,j]))**2
    return total / (height*width)

# ===== Daftar noise =====
noise_list = [
    ("Salt", "2%", 0.02, "salt"),
    ("Salt", "10%", 0.10, "salt"),
    ("Pepper", "2%", 0.02, "pepper"),
    ("Pepper", "10%", 0.10, "pepper"),
    ("Gaussian", "σ = 10", 10, "gaussian"),
    ("Gaussian", "σ = 50", 50, "gaussian")
]

# ===== Proses semua noise =====
table_data = []
for name, level, value, noise_type in noise_list:
    print(f"🔹 Memproses noise {name} {level} ...")
    if noise_type == "salt":
        noisy_rgb = add_salt(img_rgb, value)
    elif noise_type == "pepper":
        noisy_rgb = add_pepper(img_rgb, value)
    else:
        noisy_rgb = add_gaussian(img_rgb, value)

    gray_noisy = rgb2gray_manual(noisy_rgb)
    print("   - Filter Min...")
    mse_min = mse_manual(gray, min_filter_manual(gray_noisy))
    print("   - Filter Max...")
    mse_max = mse_manual(gray, max_filter_manual(gray_noisy))
    print("   - Filter Mean...")
    mse_mean = mse_manual(gray, mean_filter_manual(gray_noisy))
    print("   - Filter Median...")
    mse_median = mse_manual(gray, median_filter_manual(gray_noisy))

    table_data.append([name, level, mse_min, mse_max, mse_mean, mse_median])
    print(f"Selesai {name} {level}\n")

# ===== Cetak tabel =====
print("| Jenis Noise | Tingkat Noise | Min Filter | Max Filter | Mean Filter | Median Filter |")
print("| ----------- | ------------- | ---------- | ---------- | ----------- | ------------- |")
for row in table_data:
    print(f"| {row[0]:<11} | {row[1]:<13} | {row[2]:<10.2f} | {row[3]:<10.2f} | {row[4]:<11.2f} | {row[5]:<13.2f} |")

# Persiapan data untuk plotting
labels = [f"{row[0]} {row[1]}" for row in table_data]
min_values = [row[2] for row in table_data]
max_values = [row[3] for row in table_data]
mean_values = [row[4] for row in table_data]
median_values = [row[5] for row in table_data]

x = np.arange(len(labels))
width = 0.2

# Buat figure dengan ukuran yang lebih besar
fig, ax = plt.subplots(figsize=(14, 8))

# Buat bar chart
bars1 = ax.bar(x - 1.5*width, min_values, width, label='Min Filter', color='#FF6B6B')
bars2 = ax.bar(x - 0.5*width, max_values, width, label='Max Filter', color='#4ECDC4')
bars3 = ax.bar(x + 0.5*width, mean_values, width, label='Mean Filter', color='#45B7D1')
bars4 = ax.bar(x + 1.5*width, median_values, width, label='Median Filter', color='#96CEB4')

# Gunakan skala logaritmik untuk sumbu Y
ax.set_yscale('log')

# Tambahkan label dan judul
ax.set_xlabel('Jenis dan Tingkat Noise', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
ax.set_title('Perbandingan Performa Filter terhadap Berbagai Jenis Noise', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.02, 1), borderaxespad=0)

# Tambahkan grid untuk memudahkan pembacaan
ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')
ax.set_axisbelow(True)

# Tambahkan nilai di atas setiap bar
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)
autolabel(bars4)

plt.tight_layout()
plt.show()
