# Tugas 05: Deep Learning

Perbandingan *training from scratch*, *feature extraction*, dan *progressive fine-tuning* pada ResNet-18
dengan subset 10 kelas Oxford-IIIT Pet (5 kucing + 5 anjing, 1.996 citra), serta pembanding EfficientNet-B0.

## Struktur repositori
- `notebook.ipynb`: seluruh pipeline (data, training, evaluasi, failure analysis)
- `requirements.txt`: daftar library
- `README.md`: penjelasan

## Cara menjalankan
1. Buka notebook di Google Colab dan pilih runtime GPU (T4).
2. Jalankan **Run all**. Dataset diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet` ke folder `data/`.
3. Seluruh log tersimpan di `results/`.

## Protokol eksperimen
ResNet-18 (`IMAGENET1K_V1`), input 224x224 dengan normalisasi ImageNet, batch size 32, AdamW (weight decay 1e-2),
CrossEntropyLoss, early stopping pada validation loss (patience 5, maksimum 30 epoch), seed 42,
split stratified 70/15/15 (1.397 train / 299 val / 300 test). Pada kondisi dengan lapisan beku, modul beku
dijalankan dalam mode `eval()` agar running statistics BatchNorm bawaan tidak berubah.

## Ringkasan hasil
Sumber: `results/summary.csv` (epoch ditulis terbaik / dijalankan; waktu dalam detik; satu seed).

| Kode | Kondisi | Trainable params | Epoch | Waktu (s) | Akurasi val | Macro F1 test |
|---|---|---|---|---|---|---|
| B1 | From scratch | 11.181.642 | 12 / 17 | 243.6 | 51.17% | 0.4484 |
| B2 | Feature extraction (backbone beku) | 5.130 | 27 / 30 | 395.7 | 98.66% | 0.9668 |
| B3 | Fine-tuning `layer4` + `fc` (dengan augmentasi) | 8.398.858 | 6 / 11 | 145.1 | 97.32% | 0.9733 |
| B4 | B3 tanpa augmentasi | 8.398.858 | 4 / 9 | 86.4 | 97.32% | 0.9765 |
| B5 | EfficientNet-B0, fine-tuning 2 blok akhir + classifier (dengan augmentasi) | 1.142.202 | 9 / 14 | 186.9 | 99.00% | 0.9801 |

Analisis lengkap dan keterbatasan studi ada di laporan.
