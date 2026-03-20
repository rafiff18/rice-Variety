# Submission Klasifikasi Gambar: Rice Dataset

## Judul & Deskripsi Proyek

Proyek pengembangan model *Machine Learning* berbasis *Convolutional Neural Network* (CNN) untuk mengklasifikasikan jenis-jenis beras . Tujuan utama dari model ini adalah mengenali pola visual dari gambar beras dan mengategorikannya ke dalam 5 kelas yang berbeda (Arborio, Basmati, Ipsala, Jasmine, Karacadag). Model ini juga dikonversi ke dalam format TFLite dan TensorFlow.js (TFJS) agar siap diintegrasikan ke platform *mobile* maupun *web*.

## Dataset

Dataset yang digunakan dalam proyek ini bersumber dari Kaggle, yaitu **Rice Image Dataset** (dibuat oleh Murat Koklu). Dataset ini diunduh secara otomatis menggunakan *library* `kagglehub` (`muratkokludataset/rice-image-dataset`). Pada proyek ini,saya mengambil masing2 gambar sebanyak 300 sample per jenis.

Berikut adalah daftar *library* utama yang perlu diinstal untuk menjalankan proyek ini:

- `tensorflow==2.21.0` (Framework utama Keras/Deep Learning)
- `kagglehub==1.0.0` (Untuk mengunduh dataset langsung dari Kaggle)
- `tensorflowjs` (Untuk konversi model ke format web)
- `numpy==2.4.3` (Untuk komputasi matriks/array)
- `matplotlib==3.10.8` (Untuk visualisasi dataset dan metrik evaluasi)

**Cara Instalasi:**
Kamu bisa menginstal semua kebutuhan secara otomatis melalui file `requirements.txt`:

```bash
pip install -r requirements.txt
```


## Cara Penggunaan (Usage)

### 1. Menjalankan Notebook

* *Clone* repositori ini
* Buka file `notebook.ipynb` menggunakan Google Colab, Jupyter Notebook, atau IDE pilihanmu.
* Pastikan lingkungan (kernel) Python sudah terhubung dan dependensi sudah terinstal.

### 2. Melakukan Pelatihan (Training)

* Jalankan sel ( *cells* ) di dalam *notebook* secara berurutan.
* Alur *notebook* akan secara otomatis mengeksekusi:
  1. Pengunduhan dataset dari Kaggle ke dalam *local path* `./rice-dataset`.
  2. Pembagian dataset (Training 80%, Validation 10%, Testing 10%) menggunakan Keras Keras `image_dataset_from_directory`.
  3. Pembuatan arsitektur CNN dan pelatihan ( *training* ) menggunakan *Early Stopping* dan *Custom Callback* (otomatis berhenti jika akurasi mencapai >= 95%).
  4. Pembuatan grafik visualisasi *loss* dan  *accuracy* .

### 3. Mengeksekusi Inferensi (Prediksi TFLite)

* Bagian akhir *notebook* berisi skrip untuk menjalankan inferensi menggunakan model yang sudah dikonversi ke format `.tflite`.
* Skrip tersebut akan mengambil satu gambar secara acak dari folder dataset, memprosesnya dengan  *TFLite Interpreter* , dan menampilkan gambar uji beserta prediksi **Nama Kelas Beras** dan  **Tingkat Akurasi (Confidence %)** .
