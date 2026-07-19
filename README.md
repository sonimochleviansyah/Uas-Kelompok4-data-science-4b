# Tugas Pengganti UTS — Data Science
## Klastering Pola Konsumsi Energi Rumah Tangga untuk Efisiensi Distribusi PLN

**Kelompok 3 | Mata Kuliah Data Science**

---

## Anggota Kelompok

| NIM | Nama | Peran |
|-----|------|-------|
| 301240040 | Tegar Bagus Permana | Project Lead / PPT |
| 301240037 | Sigit Miraj Permana | ML Engineer |
| 301240041 | Selsa Shafana Alifiyani | Data Analyst |
| 301240030 | Sony Moch Leviansyah | Data Engineer |

---

## Struktur Proyek

```
Tugas_Penganti_UTS_Kelompok3/
├── data/
│   ├── raw/                        # Dataset mentah dari UCI (tidak di-commit)
│   └── processed/
│       └── clean_power_data.csv    # Data bersih hasil preprocessing
├── models/
│   ├── kmeans_model.pkl            # Model K-Means terlatih
│   └── scaler.pkl                  # Objek StandardScaler
├── notebooks/
│   └── Tugas_Penganti_UTS_Kelompok3.ipynb  # Notebook utama CRISP-DM
├── outputs/
│   └── cluster_daily_load_profile.png      # Grafik kurva beban klaster
├── src/
│   └── train_model.py              # Script pipeline training
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan pipeline training
```bash
python src/train_model.py
```

### 3. Buka notebook
```bash
jupyter notebook notebooks/Tugas_Penganti_UTS_Kelompok3.ipynb
```

---

## Dataset

- **Sumber:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Nama:** Individual Household Electric Power Consumption
- **Periode:** Desember 2006 — November 2010
- **Subset:** 100.000 observasi pertama

---

## Hasil Model

| Metrik | Nilai |
|--------|-------|
| Algoritma | K-Means Clustering |
| Jumlah Klaster (K) | 3 |
| Silhouette Score | 0.3534 |
| Interpretasi | Cukup / Reasonable Structure |
# UTS_Data-Science
