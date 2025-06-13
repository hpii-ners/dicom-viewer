# --- dicom_watcher.py ---
import os
import time
import logging
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Konfigurasi ---
BASE_DIR = os.path.dirname(__file__)
WATCH_FOLDER = os.path.join(BASE_DIR, "received_dicom")  # ✅ Benar
PROCESSED_FOLDER = os.path.join(WATCH_FOLDER, "_processed")
LOG_FILE = os.path.join(BASE_DIR, "dicom_watcher.log")
XSEG_SCRIPT = os.path.join(BASE_DIR, "dicom_xray.py")

# --- Logging ---
logger = logging.getLogger("dicom_watcher")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Pastikan folder tersedia ---
os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- Utilitas ---
def is_dicom_folder_ready(folder_path):
    """Pastikan folder mengandung file .dcm."""
    dcm_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
    return len(dcm_files) > 0

def has_been_processed(folder_name):
    return os.path.exists(os.path.join(PROCESSED_FOLDER, folder_name + ".done"))

def mark_as_processed(folder_name):
    with open(os.path.join(PROCESSED_FOLDER, folder_name + ".done"), "w") as f:
        f.write("processed\n")

def run_segmentation(folder_path):
    try:
        logger.info(f"[PROCESSING] Menjalankan segmentasi pada: {folder_path}")
        result = subprocess.run(
            ["python", XSEG_SCRIPT, folder_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"[SUCCESS] Segmentasi berhasil: {folder_path}")
        else:
            logger.error(f"[FAILED] Segmentasi gagal:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        logger.exception(f"[EXCEPTION] Kesalahan saat memproses folder: {e}")
        return False

# --- Handler Watchdog ---
class DICOMFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            folder_path = event.src_path
            folder_name = os.path.basename(folder_path)

            logger.info(f"[DETECTED] Folder baru: {folder_name}")
            time.sleep(3)  # Tunggu file ditulis

            if has_been_processed(folder_name):
                logger.info(f"[SKIP] Folder {folder_name} sudah diproses.")
                return

            if is_dicom_folder_ready(folder_path):
                success = run_segmentation(folder_path)
                if success:
                    mark_as_processed(folder_name)
            else:
                logger.warning(f"[SKIP] {folder_name} tidak mengandung file .dcm")

# --- Main loop ---
def main():
    logger.info("🔎 Memulai pemantauan folder DICOM...")
    event_handler = DICOMFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_FOLDER, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("👋 Watcher dihentikan oleh pengguna.")
    observer.join()

if __name__ == "__main__":
    main()
