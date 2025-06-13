import os
import sys
import torch
import logging
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged, # Menggunakan ScaleIntensityRanged untuk kontrol lebih baik
    ResizeWithPadOrCropd,
    ToTensord,
)
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset, NibabelReader # Penting: Impor NibabelReader
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from PIL import Image # Untuk menyimpan gambar hasil
from datetime import datetime
from skimage.transform import resize
from skimage.measure import regionprops, label # Untuk analisis region


# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dicom_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dicom_xray_analyzer")

# --- Konfigurasi ---
# Pastikan ini nama file model SwinUNETR Anda yang sudah dilatih
MODEL_PATH = "xray_segmentation_model.pt" # Menggunakan nama model yang konsisten
# Ukuran gambar yang diinginkan oleh model SwinUNETR Anda (misal: ukuran patch saat training)
# Ini BUKAN ukuran asli gambar, tapi ukuran input yang diharapkan oleh model.
# Sesuaikan ini dengan `img_size` saat melatih model Anda.
MODEL_INPUT_SIZE = (96, 96) # Contoh: Jika model dilatih dengan patch 96x96
# Ukuran ROI untuk sliding window inference, sebaiknya sama dengan MODEL_INPUT_SIZE
SWINUNETR_ROI_SIZE = MODEL_INPUT_SIZE

OUTPUT_FOLDER = "segmentation_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Fungsi: Load DICOM dengan Metadata ---
def load_dicom_data(dicom_folder):
    """
    Memuat file DICOM utama dari folder dan mengekstrak metadata penting.
    Mengembalikan jalur ke file DICOM, bukan gambar sementara.
    """
    try:
        dcm_files = [f for f in os.listdir(dicom_folder) if f.lower().endswith(".dcm")]
        if not dcm_files:
            raise FileNotFoundError(f"Tidak ditemukan file DICOM (.dcm) di folder: {dicom_folder}")

        # Ambil file DICOM pertama (seringkali ada satu per studi X-ray)
        dicom_filename = sorted(dcm_files)[0]
        dicom_path = os.path.join(dicom_folder, dicom_filename)
        ds = pydicom.dcmread(dicom_path)
        
        # Ekstrak metadata
        metadata = {
            'PatientID': getattr(ds, 'PatientID', 'N/A'),
            'PatientName': str(getattr(ds, 'PatientName', 'N/A')),
            'StudyDate': getattr(ds, 'StudyDate', 'N/A'),
            'Modality': getattr(ds, 'Modality', 'N/A'),
            'BodyPartExamined': getattr(ds, 'BodyPartExamined', 'N/A'),
            'StudyDescription': getattr(ds, 'StudyDescription', 'N/A'),
            'Rows': getattr(ds, 'Rows', 'N/A'),
            'Columns': getattr(ds, 'Columns', 'N/A'),
            'PixelSpacing': getattr(ds, 'PixelSpacing', 'N/A')
        }
        logger.info(f"Metadata DICOM dari {dicom_filename}: {metadata}")
        
        # Original size dari pixel_array
        original_size = ds.pixel_array.shape # (H, W)
        
        return dicom_path, original_size, os.path.splitext(dicom_filename)[0], metadata
    
    except Exception as e:
        logger.error(f"Gagal memuat data DICOM dari {dicom_folder}: {str(e)}", exc_info=True)
        raise

# --- Fungsi: Get Transforms ---
def get_transforms(target_size):
    """
    Mendapatkan transformasi data untuk preprocessing gambar X-ray.
    Transformasi ini harus sesuai dengan yang digunakan saat pelatihan model.
    """
    return Compose([
        # LoadImaged dengan NibabelReader (kompatibel dengan PydicomReader internal MONAI untuk DICOM)
        LoadImaged(keys=["image"], reader=NibabelReader()), # Menggunakan NibabelReader() untuk DICOM
        EnsureChannelFirstd(keys=["image"]),
        # ScaleIntensityRanged: Sesuaikan rentang ini dengan preprocessing pelatihan model Anda
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=1000, b_min=0.0, b_max=1.0, clip=True), 
        ResizeWithPadOrCropd(keys=["image"], spatial_size=target_size, mode="constant"),
        ToTensord(keys=["image"]),
    ])

# --- Fungsi: Load Model ---
def load_model(device, img_size=MODEL_INPUT_SIZE):
    """Memuat model SwinUNETR"""
    logger.info(f"Memuat model SwinUNETR dari {MODEL_PATH} dengan ukuran {img_size}...")
    
    try:
        model = SwinUNETR(
            img_size=img_size,
            spatial_dims=2,
            in_channels=1,
            out_channels=1, # Jika ini segmentasi biner (misal: paru-paru vs background)
            feature_size=48,
            use_checkpoint=True, # Set ke True jika Anda menggunakan checkpointing saat training
        ).to(device)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan di: {MODEL_PATH}. Pastikan nama dan lokasi benar.")

        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Penanganan state_dict: kadang model disimpan dengan 'module.' prefix atau di bawah 'state_dict' kunci
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Hapus 'module.' prefix jika model dilatih dengan DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            logger.info("Model state_dict dimuat dari kunci 'state_dict'.")
        else:
            model.load_state_dict(checkpoint) # Asumsi checkpoint adalah state_dict langsung
            logger.info("Model state_dict dimuat langsung.")
            
        model.eval() # Set model ke mode evaluasi
        logger.info("Model berhasil dimuat dan disetel ke mode evaluasi.")
        return model
        
    except Exception as e:
        logger.error(f"Gagal memuat model dari {MODEL_PATH}: {str(e)}", exc_info=True)
        sys.exit(1) # Keluar jika model tidak bisa dimuat

# --- Fungsi: Save Segmentation Results ---
def save_segmentation_results(mask_array, original_dicom_path, output_base_filename, original_size):
    """
    Menyimpan hasil segmentasi dalam berbagai format (mask, overlay, comparison).
    Menggunakan gambar asli dari DICOM.
    """
    try:
        ds = pydicom.dcmread(original_dicom_path)
        original_pixel_array = ds.pixel_array.astype(np.float32)

        # Normalisasi untuk visualisasi (jika pixel_array adalah gambar asli dengan rentang intensitas penuh)
        original_pixel_array_norm = (original_pixel_array - original_pixel_array.min()) / \
                                    (original_pixel_array.max() - original_pixel_array.min())
        
        # Resize mask ke ukuran asli gambar
        if mask_array.shape != original_size:
            logger.info(f"Mengubah ukuran mask dari {mask_array.shape} ke ukuran asli {original_size}")
            mask_array = resize(mask_array, original_size, preserve_range=True, anti_aliasing=True)
        
        # Konversi mask ke biner (0 atau 1) dan ke 0-255 untuk visualisasi PNG
        binary_mask_for_viz = (mask_array > 0.5).astype(np.uint8) * 255
        
        # Buat objek Image dari numpy array
        original_img_pil = Image.fromarray((original_pixel_array_norm * 255).astype(np.uint8)).convert("L")
        mask_img_pil = Image.fromarray(binary_mask_for_viz).convert("L")
        
        # Gabungkan gambar asli dengan mask (overlay)
        # Pastikan kedua gambar adalah RGB untuk blend
        overlay_pil = Image.blend(
            original_img_pil.convert("RGB"),
            Image.fromarray(np.dstack([mask_img_pil]*3)).convert("RGB"), # RGB mask
            alpha=0.4 # Transparansi mask
        )
        
        # Simpan semua versi
        # base_name_for_output akan menjadi nama folder DICOM atau ID studi
        base_name_for_output = output_base_filename 
        
        # 1. Mask saja
        mask_img_pil.save(os.path.join(OUTPUT_FOLDER, f"{base_name_for_output}_mask.png"))
        
        # 2. Overlay
        overlay_pil.save(os.path.join(OUTPUT_FOLDER, f"{base_name_for_output}_overlay.png"))
        
        # 3. Side-by-side comparison
        # Pastikan ukuran untuk gambar gabungan
        combined_width = original_size[1] * 2 # Dua gambar berdampingan
        combined_height = original_size[0]
        
        combined_pil = Image.new("RGB", (combined_width, combined_height))
        combined_pil.paste(original_img_pil.convert("RGB"), (0, 0))
        combined_pil.paste(overlay_pil, (original_size[1], 0)) # overlay ditempatkan di sebelah kanan
        combined_pil.save(os.path.join(OUTPUT_FOLDER, f"{base_name_for_output}_comparison.png"))
        
        logger.info(f"Hasil segmentasi visual disimpan di: {OUTPUT_FOLDER}/{base_name_for_output}_*.png")
    
    except Exception as e:
        logger.error(f"Gagal menyimpan hasil segmentasi visual: {str(e)}", exc_info=True)
        raise

# --- Fungsi: Generate Medical Analysis ---
def generate_medical_analysis(mask_array, metadata, base_filename, original_size):
    """Membuat laporan analisis medis dan mencatat ke log/file."""
    try:
        # Resize mask ke ukuran asli untuk analisis metrik yang akurat
        if mask_array.shape != original_size:
            mask_array_for_analysis = resize(mask_array, original_size, preserve_range=True, anti_aliasing=True)
        else:
            mask_array_for_analysis = mask_array

        binary_mask = (mask_array_for_analysis > 0.5).astype(np.uint8)
        total_pixels = binary_mask.size
        segmented_pixels = np.sum(binary_mask)
        
        percentage = (segmented_pixels / total_pixels) * 100
        
        # Analisis region (misal: untuk mendeteksi anomali terpisah)
        labeled_mask = label(binary_mask)
        regions = regionprops(labeled_mask)
        
        num_regions = len(regions)
        areas = [r.area for r in regions]
        max_area = max(areas) if areas else 0
        avg_area = np.mean(areas) if areas else 0
        
        # Kriteria interpretasi klinis (contoh sederhana, sesuaikan dengan kebutuhan medis)
        severity = "Tidak Diketahui"
        recommendation = "Diperlukan peninjauan lebih lanjut oleh profesional medis."

        # Contoh kriteria sederhana untuk paru-paru
        if metadata.get('BodyPartExamined') == 'CHEST' or metadata.get('StudyDescription') == 'CHEST X-RAY':
            if percentage < 5:
                severity = "Normal (Luas Area Tersegmentasi Rendah)"
                recommendation = "Tidak ada indikasi anomali signifikan berdasarkan segmentasi otomatis."
            elif 5 <= percentage < 15:
                severity = "Ringan (Perlu Perhatian)"
                recommendation = "Area tersegmentasi menunjukkan variasi kecil. Pemantauan rutin disarankan."
            elif 15 <= percentage < 30:
                severity = "Sedang (Perlu Konsultasi)"
                recommendation = "Area tersegmentasi signifikan. Konsultasi dengan spesialis radiologi disarankan untuk diagnosis."
            else:
                severity = "Parah (Evaluasi Mendesak)"
                recommendation = "Area tersegmentasi sangat besar. Diperlukan evaluasi dan intervensi medis segera."
        else:
             if percentage < 5:
                severity = "Normal (Luas Area Tersegmentasi Rendah)"
                recommendation = "Tidak ada indikasi anomali signifikan berdasarkan segmentasi otomatis."
             elif 5 <= percentage < 15:
                severity = "Ringan (Perlu Perhatian)"
                recommendation = "Area tersegmentasi menunjukkan variasi kecil. Pemantauan rutin disarankan."
             elif 15 <= percentage < 30:
                severity = "Sedang (Perlu Konsultasi)"
                recommendation = "Area tersegmentasi signifikan. Konsultasi dengan spesialis radiologi disarankan untuk diagnosis."
             else:
                severity = "Parah (Evaluasi Mendesak)"
                recommendation = "Area tersegmentasi sangat besar. Diperlukan evaluasi dan intervensi medis segera."

        report_lines = [
            "\n" + "="*30,
            "=== LAPORAN ANALISIS MEDIS DICOM X-RAY ===",
            "="*30,
            f"Tanggal Analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n--- INFORMASI PASIEN ---",
            f"ID Pasien: {metadata.get('PatientID', 'N/A')}",
            f"Nama Pasien: {metadata.get('PatientName', 'N/A')}",
            f"Tanggal Pemeriksaan: {metadata.get('StudyDate', 'N/A')}",
            f"Modality: {metadata.get('Modality', 'N/A')}",
            f"Area Pemeriksaan: {metadata.get('BodyPartExamined', 'N/A')}",
            f"Deskripsi Pemeriksaan: {metadata.get('StudyDescription', 'N/A')}",
            f"Ukuran Gambar Asli (Px): {original_size[1]}x{original_size[0]}",
            "\n--- HASIL SEGMENTASI ---",
            f"Ukuran Mask Segmentasi (Px): {binary_mask.shape[1]}x{binary_mask.shape[0]}",
            f"Total Piksel Gambar: {total_pixels} piksel",
            f"Luas Area Tersegmentasi: {segmented_pixels} piksel",
            f"Persentase Area Tersegmentasi: {percentage:.2f}%",
            f"Jumlah Region Terdeteksi: {num_regions}",
            f"Area Maksimum Region: {max_area:.2f} piksel",
            f"Area Rata-rata Region: {avg_area:.2f} piksel",
            "\n--- INTERPRETASI KLINIS ---",
            f"Tingkat Keparahan Prediksi: {severity}",
            f"Rekomendasi Awal: {recommendation}",
            "\n--- CATATAN TEKNIS ---",
            "Segmentasi dilakukan menggunakan model AI SwinUNETR yang dilatih khusus.",
            "Hasil ini adalah alat bantu dan harus ditinjau oleh profesional medis yang berkualifikasi.",
            "Model AI bukan pengganti penilaian klinis dan diagnosa profesional."
        ]
        
        # Cetak ke konsol dan log file
        for line in report_lines:
            logger.info(line)
        
        # Simpan ke file teks
        text_output_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_analysis.txt")
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Laporan analisis disimpan di: {text_output_path}")
        
    except Exception as e:
        logger.error(f"Gagal membuat laporan analisis: {str(e)}", exc_info=True)
        raise

# --- Main Function ---
def main():
    if len(sys.argv) < 2:
        logger.error("Penggunaan: python dicom_xray.py <folder_dicom>")
        sys.exit(1)

    dicom_folder = sys.argv[1]
    if not os.path.isdir(dicom_folder):
        logger.error(f"Folder DICOM tidak ditemukan: {dicom_folder}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Menggunakan perangkat: {device}")
    set_determinism(42) # Untuk reproduksibilitas

    try:
        # 1. Load DICOM Data dan Metadata
        logger.info(f"Memproses folder DICOM: {dicom_folder}")
        dicom_file_path, original_size, base_filename, metadata = load_dicom_data(dicom_folder)
        logger.info(f"DICOM file path: {dicom_file_path}")
        logger.info(f"Ukuran gambar asli: {original_size[1]}x{original_size[0]}") # Kolom x Baris

        # 2. Load Model
        # Pastikan MODEL_INPUT_SIZE sesuai dengan img_size model Anda
        model = load_model(device, img_size=MODEL_INPUT_SIZE) 
        
        # 3. Persiapkan Data untuk Inferensi
        data_dict = [{"image": dicom_file_path}]
        # Gunakan MODEL_INPUT_SIZE sebagai target_size untuk transformasi
        transforms = get_transforms(target_size=MODEL_INPUT_SIZE)
        dataset = Dataset(data=data_dict, transform=transforms)
        loader = DataLoader(dataset, batch_size=1, num_workers=0) # num_workers=0 untuk debugging

        # 4. Lakukan Inferensi (Segmentasi)
        logger.info("Menjalankan inferensi (segmentasi)...")
        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(device)
                logger.info(f"Ukuran input ke model setelah transform: {inputs.shape}")
                
                # roi_size untuk sliding window inference, sebaiknya sama dengan MODEL_INPUT_SIZE
                sw_roi_size = SWINUNETR_ROI_SIZE 
                logger.info(f"Menggunakan ROI size untuk sliding window inference: {sw_roi_size}")
                
                # sw_batch_size harus diatur agar sesuai dengan kapasitas GPU Anda.
                # Jika input_size besar, ini bisa menyebabkan OOM error.
                outputs = sliding_window_inference(
                    inputs, 
                    roi_size=sw_roi_size,
                    sw_batch_size=1, # Jika gambar X-ray 2D, ini seringkali 1 atau kecil
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )
                
                # Menggunakan sigmoid karena out_channels=1, untuk mendapatkan probabilitas 0-1
                mask_tensor = torch.sigmoid(outputs)
                mask_array_raw = mask_tensor.cpu().numpy()[0][0] # Ambil array NumPy (H, W)

                # Hasil mask perlu diresize kembali ke ukuran asli DICOM untuk visualisasi dan analisis
                # Fungsi save_segmentation_results dan generate_medical_analysis akan menangani resizing ini.
                
                break # Hanya memproses satu gambar per folder

        # 5. Simpan Hasil Visual
        save_segmentation_results(mask_array_raw, dicom_file_path, base_filename, original_size)
        
        # 6. Buat Laporan Analisis Medis
        generate_medical_analysis(mask_array_raw, metadata, base_filename, original_size)
        
        logger.info("Proses analisis DICOM X-ray selesai dengan sukses!")
        
    except Exception as e:
        logger.error(f"Gagal memproses folder DICOM '{dicom_folder}': {str(e)}", exc_info=True)
        sys.exit(1) # Keluar dengan kode error

if __name__ == "__main__":
    main()