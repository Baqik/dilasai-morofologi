# dilasai-morofologi
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: Load gambar tulisan tangan ---
image_path = 's_Prasasti.png'  # Ganti dengan path gambar kamu
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Cek apakah gambar berhasil dibaca
if image is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Periksa kembali path atau nama file-nya.")

# Binarisasi gambar
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# --- STEP 2: Buat 9 Strel (Structuring Element) ---
def generate_strels():
    sizes = [3, 5, 7]
    shapes = {
        'Rect': cv2.MORPH_RECT,
        'Ellipse': cv2.MORPH_ELLIPSE,
        'Cross': cv2.MORPH_CROSS
    }
    strel_list = []
    for size in sizes:
        for name, shape in shapes.items():
            kernel = cv2.getStructuringElement(shape, (size, size))
            strel_list.append((f"{name}_{size}x{size}", kernel))
    return strel_list

strels = generate_strels()

# --- STEP 3: Lakukan operasi erosi dan dilasi ---
results = []
for name, kernel in strels:
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    results.append((name, kernel.shape, eroded, dilated))

# --- STEP 4: Tampilkan hasilnya ---
rows = len(results)
plt.figure(figsize=(12, rows * 2))

for idx, (name, shape, eroded, dilated) in enumerate(results):
    plt.subplot(rows, 3, idx*3 + 1)
    plt.imshow(binary, cmap='gray')
    plt.title(f'Asli ({name})')
    plt.axis('off')
    
    plt.subplot(rows, 3, idx*3 + 2)
    plt.imshow(eroded, cmap='gray')
    plt.title(f'Erosi ({name})')
    plt.axis('off')
    
    plt.subplot(rows, 3, idx*3 + 3)
    plt.imshow(dilated, cmap='gray')
    plt.title(f'Dilasi ({name})')
    plt.axis('off')

plt.tight_layout()
plt.show()
