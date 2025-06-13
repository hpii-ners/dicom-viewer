# dicom_server.py
import os
import logging
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, StoragePresentationContexts
from pydicom.uid import (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    RLELossless
)

import configparser
from db_handler import save_metadata, get_str # Import get_str dari db_handler untuk konsistensi

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# --- Load Konfigurasi dari config.ini ---
config = configparser.ConfigParser()
try:
    config.read('config.ini')
    if 'dicom' not in config:
        raise ValueError("Bagian '[dicom]' tidak ditemukan di config.ini")
except Exception as e:
    logging.error(f"Gagal membaca config.ini atau bagian '[dicom]' tidak ada: {e}")
    logging.error("Pastikan Anda memiliki file config.ini dengan bagian [dicom] yang benar.")
    exit(1)

server_config = config['dicom']
ae_title = server_config.get('aet', 'FLASK_DICOM_SCP').encode('utf-8')
ip = server_config.get('ip', '0.0.0.0')
port = server_config.getint('port', 11112)
logging.info(f"Konfigurasi dimuat: AE Title='{ae_title.decode()}', IP='{ip}', Port='{port}'")


# Pastikan RECEIVED_DIR relatif terhadap direktori skrip utama
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RECEIVED_DIR = os.path.join(BASE_DIR, 'received_dicom')
os.makedirs(RECEIVED_DIR, exist_ok=True)
logging.info(f"File DICOM akan disimpan di: {os.path.abspath(RECEIVED_DIR)}")

def handle_store(event):
    """
    Menangani request C-STORE yang diterima.
    Menyimpan file DICOM dan metadata ke database.
    """
    try:
        ds = event.dataset
        ds.file_meta = event.file_meta

        # Dapatkan AccessionNumber untuk membuat folder.
        # Gunakan 'UNKNOWN_ACCESSION' sebagai fallback jika tidak ada.
        accession_number = get_str(ds.get("AccessionNumber", "UNKNOWN_ACCESSION"))
        # Sanitisasi AccessionNumber untuk nama folder yang aman
        accession_number_safe = "".join(c for c in accession_number if c.isalnum() or c in ('-', '_')).strip()
        if not accession_number_safe: # Jika setelah sanitasi kosong, gunakan SOPInstanceUID
            accession_number_safe = str(ds.SOPInstanceUID)
            logging.warning(f"AccessionNumber kosong atau tidak valid, menggunakan SOPInstanceUID sebagai nama folder: {accession_number_safe}")
            
        # Buat direktori berdasarkan AccessionNumber
        study_dir = os.path.join(RECEIVED_DIR, accession_number_safe)
        os.makedirs(study_dir, exist_ok=True)
        logging.debug(f"Memastikan direktori studi ada: {study_dir}")

        # Tentukan jalur absolut untuk menyimpan file DICOM
        filepath_absolute = os.path.join(study_dir, f"{ds.SOPInstanceUID}.dcm")
        
        # Simpan file DICOM secara fisik
        ds.save_as(filepath_absolute, write_like_original=False)
        logging.info(f"Berhasil menerima dan menyimpan file DICOM: {filepath_absolute}")

        try:
            # Hitung jalur relatif dari RECEIVED_DIR untuk disimpan di database
            # Ini akan menjadi sesuatu seperti '1059953627/1.2.840....dcm'
            filepath_relative_for_db = os.path.relpath(filepath_absolute, RECEIVED_DIR)
            
            # Panggil save_metadata dengan jalur relatif
            save_metadata(filepath_relative_for_db) 
            logging.info(f"Metadata untuk {filepath_absolute.split(os.sep)[-1]} berhasil disimpan ke database menggunakan jalur relatif: {filepath_relative_for_db}")
        except Exception as e:
            logging.error(f"Gagal menyimpan metadata DICOM ke database untuk {filepath_absolute.split(os.sep)[-1]}: {e}")
            return 0xC001 # C-STORE Status: Failure (Processing Failure)

        return 0x0000 # C-STORE Status: Success
    except Exception as e:
        logging.error(f"Terjadi exception tak tertangani saat memproses C-STORE request: {e}")
        return 0xC210 # C-STORE Status: Failure (Cannot understand)

handlers = [(evt.EVT_C_STORE, handle_store)]

ae = AE(ae_title=ae_title)

# --- KONFIGURASI KRUSIAL UNTUK KOMPATIBILITAS MAKSIMAL ---
# Daftarkan semua Transfer Syntax yang ingin didukung.
transfer_syntaxes_to_support = [
    ImplicitVRLittleEndian,     # Default Transfer Syntax
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian
]

logging.info("Menambahkan konteks presentasi untuk semua SOP Class Storage dengan Transfer Syntax umum.")
# Tambahkan semua konteks presentasi penyimpanan standar dengan semua transfer syntax yang didukung
for context in StoragePresentationContexts:
    ae.add_supported_context(context.abstract_syntax, transfer_syntaxes_to_support)

logging.info("Konteks Presentasi yang Didukung:")
for cx in ae.supported_contexts:
    logging.info(f"  - SOP Class: {cx.abstract_syntax.name} ({str(cx.abstract_syntax)})")
    logging.info(f"    Transfer Syntaxes: {[ts.name for ts in cx.transfer_syntax]}")


if __name__ == "__main__":
    try:
        logging.info(f"Memulai server DICOM di {ip}:{port} dengan AE Title '{ae_title.decode()}'")
        ae.start_server((ip, port), evt_handlers=handlers, block=True)
    except OSError as e:
        logging.error(f"Gagal memulai server DICOM: {e}")
        logging.error(f"Pastikan port {port} tidak digunakan oleh aplikasi lain dan Anda memiliki izin yang diperlukan.")
    except Exception as e:
        logging.error(f"Terjadi kesalahan tidak terduga saat menjalankan server: {e}")

