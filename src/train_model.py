import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

for directory in [DATA_RAW_DIR, DATA_PROC_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)


def load_and_extract_data():
    print("[1/6] Memeriksa keberadaan dataset...")
    txt_path = os.path.join(DATA_RAW_DIR, 'household_power_consumption.txt')
    
    if not os.path.exists(txt_path):
        print("   -> ERROR: File tidak ditemukan di data/raw/")
        print("   -> Pastikan household_power_consumption.txt ada di folder data/raw/")
        raise FileNotFoundError(f"Dataset tidak ditemukan: {txt_path}")
    
    print("   -> Dataset ditemukan di data/raw/")
    return txt_path


def preprocess_data(file_path):
    print("[2/6] Memulai data preprocessing...")
    df = pd.read_csv(
        file_path,
        sep=';',
        na_values=['?'],
        low_memory=False,
        nrows=100000
    )

    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)

    df.dropna(inplace=True)

    num_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col])

    df['Hour'] = df.index.hour
    df['Is_Weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    proc_path = os.path.join(DATA_PROC_DIR, 'clean_power_data.csv')
    df.to_csv(proc_path)
    print(f"   -> Data bersih disimpan: {df.shape[0]:,} baris, {df.shape[1]} kolom")
    print(f"   -> Lokasi: {proc_path}")

    return df, num_cols


def scale_features(df, features):
    print("[3/6] Melakukan feature scaling...")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[features])
    df_scaled = pd.DataFrame(scaled_array, columns=features, index=df.index)

    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"   -> Scaler disimpan di: {scaler_path}")

    return df_scaled


def train_kmeans(df_scaled, n_clusters=3):
    print(f"[4/6] Melatih model K-Means dengan K={n_clusters}...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    labels = kmeans.fit_predict(df_scaled)

    sample_size = min(10000, df_scaled.shape[0])
    sample_idx = np.random.choice(df_scaled.shape[0], sample_size, replace=False)
    sil_score = silhouette_score(df_scaled.iloc[sample_idx], labels[sample_idx])
    print(f"   -> Silhouette Score: {sil_score:.4f}")

    model_path = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
    joblib.dump(kmeans, model_path)
    print(f"   -> Model disimpan di: {model_path}")

    return labels


def generate_insights(df, labels, features):
    print("[5/6] Menghasilkan profil klaster dan visualisasi...")
    df['Cluster'] = labels

    cluster_profile = df.groupby('Cluster')[features].mean()
    print("\n--- PROFIL KONSUMSI RATA-RATA PER KLASTER ---")
    print(cluster_profile.round(4))
    print("---------------------------------------------\n")

    mean_power = df.groupby('Cluster')['Global_active_power'].mean()
    sorted_clusters = mean_power.sort_values()
    label_map = {}
    for i, (cluster_id, _) in enumerate(sorted_clusters.items()):
        if i == 0:
            label_map[cluster_id] = f'Cluster {cluster_id} (Rendah / Base Load)'
        elif i == 1:
            label_map[cluster_id] = f'Cluster {cluster_id} (Menengah / Konstan)'
        else:
            label_map[cluster_id] = f'Cluster {cluster_id} (Tinggi / Peak Load)'

    print("--- DISTRIBUSI ANGGOTA PER KLASTER ---")
    for cluster_id in sorted(df['Cluster'].unique()):
        count = (df['Cluster'] == cluster_id).sum()
        pct = count / len(df) * 100
        print(f"   {label_map[cluster_id]}: {count:,} observasi ({pct:.1f}%)")
    print("---------------------------------------\n")

    colors_cluster = {0: 'steelblue', 1: 'darkorange', 2: 'firebrick'}

    plt.figure(figsize=(11, 5))
    for cluster_id in sorted(df['Cluster'].unique()):
        subset = df[df['Cluster'] == cluster_id]
        hourly = subset.groupby('Hour')['Global_active_power'].mean()
        plt.plot(
            hourly.index,
            hourly.values,
            marker='o',
            color=colors_cluster[cluster_id],
            label=label_map[cluster_id],
            linewidth=2.5
        )

    plt.axvspan(18, 22, alpha=0.08, color='red', label='Zona beban puncak PLN')
    plt.title('Profil Konsumsi Daya Aktif Harian per Klaster Pelanggan')
    plt.xlabel('Jam dalam Sehari (0-23)')
    plt.ylabel('Rata-rata Global Active Power (kW)')
    plt.xticks(range(0, 24))
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'cluster_daily_load_profile.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Visualisasi disimpan di: {plot_path}")


def run_pipeline():
    print("=" * 55)
    print("  PIPELINE TRAINING — KLASTERING KONSUMSI ENERGI PLN")
    print("=" * 55)
    print()

    data_path = load_and_extract_data()
    df_clean, features = preprocess_data(data_path)
    df_scaled = scale_features(df_clean, features)
    cluster_labels = train_kmeans(df_scaled, n_clusters=3)
    generate_insights(df_clean, cluster_labels, features)

    print("[6/6] Selesai. Seluruh output tersimpan di folder masing-masing.")


if __name__ == "__main__":
    run_pipeline()