import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Membaca citra asli 
img = cv2.imread('jalan.jpg')

# Cek apakah gambar berhasil dibaca
if img is None:
    print("Error: Gambar tidak ditemukan! Pastikan path file benar.")
else:
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Grayscale manual
    height, width, _ = img_rgb.shape
    gray = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            R = img_rgb[i, j, 0]
            G = img_rgb[i, j, 1]
            B = img_rgb[i, j, 2]
            gray[i, j] = int(0.299 * R + 0.587 * G + 0.114 * B)

    # 3. Salt noise sedikit (2%) 
    amount_salt_few = 0.02
    gray_salt_few = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            gray_salt_few[i, j] = gray[i, j]
            if np.random.rand() < amount_salt_few:
                gray_salt_few[i, j] = 255  # Salt = putih (255)

    # 4. Salt noise banyak (10%) 
    amount_salt_many = 0.1
    gray_salt_many = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            gray_salt_many[i, j] = gray[i, j]
            if np.random.rand() < amount_salt_many:
                gray_salt_many[i, j] = 255  # Salt = putih (255)

    # 5. Pepper noise sedikit (2%)
    amount_pepper_few = 0.02
    gray_pepper_few = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            gray_pepper_few[i, j] = gray[i, j]
            if np.random.rand() < amount_pepper_few:
                gray_pepper_few[i, j] = 0  # Pepper = hitam (0)

    # 6. Pepper noise banyak (10%) 
    amount_pepper_many = 0.1
    gray_pepper_many = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            gray_pepper_many[i, j] = gray[i, j]
            if np.random.rand() < amount_pepper_many:
                gray_pepper_many[i, j] = 0  # Pepper = hitam (0)

    # 7. Gaussian noise rendah (sigma=10) 
    mean_low = 0
    sigma_low = 10
    gray_gaussian_low = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            noise = np.random.normal(mean_low, sigma_low)
            new_value = gray[i, j] + noise
            if new_value < 0:
                new_value = 0
            elif new_value > 255:
                new_value = 255
            gray_gaussian_low[i, j] = int(new_value)

    # 8. Gaussian noise tinggi (sigma=50)
    mean_high = 0
    sigma_high = 50
    gray_gaussian_high = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            noise = np.random.normal(mean_high, sigma_high)
            new_value = gray[i, j] + noise
            if new_value < 0:
                new_value = 0
            elif new_value > 255:
                new_value = 255
            gray_gaussian_high[i, j] = int(new_value)

    # 9. Tampilan 5x2
    plt.figure(figsize=(14, 28))

    plt.subplot(5, 2, 1)
    plt.title("Citra Asli", fontsize=16)
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(5, 2, 2)
    plt.title("Grayscale", fontsize=16)
    plt.imshow(gray, cmap='gray')
    plt.axis("off")

    plt.subplot(5, 2, 3)
    plt.title("Salt Noise Sedikit (2%)", fontsize=16)
    plt.imshow(gray_salt_few, cmap='gray')
    plt.axis("off")

    plt.subplot(5, 2, 4)
    plt.title("Salt Noise Banyak (10%)", fontsize=16)
    plt.imshow(gray_salt_many, cmap='gray')
    plt.axis("off")

    plt.subplot(5, 2, 5)
    plt.title("Pepper Noise Sedikit (2%)", fontsize=16)
    plt.imshow(gray_pepper_few, cmap='gray')
    plt.axis("off")

    plt.subplot(5, 2, 6)
    plt.title("Pepper Noise Banyak (10%)", fontsize=16)
    plt.imshow(gray_pepper_many, cmap='gray')
    plt.axis("off")

    plt.subplot(5, 2, 7)
    plt.title("Gaussian Noise Rendah (σ=10)", fontsize=16)
    plt.imshow(gray_gaussian_low, cmap='gray')
    plt.axis("off")

    plt.subplot(5, 2, 8)
    plt.title("Gaussian Noise Tinggi (σ=50)", fontsize=16)
    plt.imshow(gray_gaussian_high, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()
