# Tugas Model Generatif

Implementasi modular dengan PyTorch untuk perbandingan model generatif pada
Fashion-MNIST. Notebook tetap tersedia sebagai panduan pembelajaran; modul
Python menyediakan alur pelatihan berbasis skrip yang dapat direproduksi.

## Struktur proyek

```text
data.py
config.py
metrics.py
models/
  autoencoder.py
  vae.py
  gan.py
  ddpm.py
  mlp_classifier.py
training.py
train_mlp.py
train_autoencoders.py
train_gans.py
train_ddpm.py
experiments/
  mlp_limits.py
evaluate.py
requirements.txt
```

## Instalasi

Buat virtual environment dan pasang dependensi. Pasang terlebih dahulu versi
PyTorch/torchvision yang sesuai dengan perangkat keras melalui
[pemilih resmi PyTorch](https://pytorch.org/get-started/locally/), lalu pasang
paket lainnya:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

Dataset Fashion-MNIST akan diunduh otomatis ke `data/`, lalu dibagi agar
dapat direproduksi dengan komposisi 80% data latih, 10% data validasi, dan 10% data uji.
Ukuran batch adalah 128; hanya data latih yang diacak.

## Urutan pelatihan

Jalankan perintah berikut dari direktori utama proyek:

```powershell
python train_mlp.py
python train_autoencoders.py
python train_gans.py
python train_ddpm.py
python -m experiments.mlp_limits
python evaluate.py
```

Eksperimen keterbatasan MLP memerlukan checkpoint MLP dan Vanilla AE, jadi
latih kedua model tersebut terlebih dahulu. Checkpoint, grafik, dan metrik
disimpan di dalam `outputs/`. Jumlah epoch dapat dikurangi untuk uji singkat,
misalnya `python train_mlp.py --epochs 1`.

Perhitungan FID membatasi penggunaan memori dengan menghasilkan paling banyak
64 gambar pada setiap iterasi; jumlah sampel evaluasi bawaan adalah 5.000.
Sampling DDPM membutuhkan komputasi lebih besar karena proses reverse
dijalankan untuk setiap sampel.
