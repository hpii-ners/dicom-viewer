# dicom_viewer.py
import os
import logging
import numpy as np
from io import BytesIO
from PIL import Image
import pydicom
from pydicom import dcmread, multival 
import pydicom.errors
from flask import (
    Flask,
    render_template, 
    request,
    send_file,
    url_for,
    abort,
    redirect,
    flash 
)
import configparser
import psycopg2 

# --- Import for Image Enhancement ---
from skimage import exposure # Untuk CLAHE
# --- End Import for Image Enhancement ---

from psycopg2.extras import DictCursor

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Tambahkan secret key untuk flash messages

# --- Load Konfigurasi dari config.ini ---
config = configparser.ConfigParser()
try:
    config.read('config.ini')
    
    # Ambil konfigurasi database
    db_config = config['database']
    
    # Ambil konfigurasi server Flask (app)
    app_host = config.get('server', 'host', fallback='0.0.0.0')
    app_port = config.getint('server', 'port', fallback=5000)
    
except Exception as e:
    logging.error(f"Error reading config.ini: {e}")
    logging.error("Please ensure your config.ini has [database] and [server] sections.")
    exit(1)

# Konfigurasi Database
DB_CONFIG = {
    "host": db_config.get("host", "localhost"),
    "database": db_config.get("database", "postgres"),
    "user": db_config.get("user", "postgres"),
    "password": db_config.get("password", "12345678"),
    "port": db_config.getint("port", 5432)
}

# Dapatkan root direktori proyek
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Definisikan RECEIVED_DICOM_ROOT secara kanonik (realpath)
RECEIVED_DICOM_ROOT = os.path.realpath(os.path.join(BASE_DIR, 'received_dicom'))


# --- Koneksi Database ---
def get_db_connection():
    """Membuka koneksi ke database PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        logging.error(f"Error connecting to database: {e}")
        # Gunakan Flask abort untuk menampilkan pesan error ke user
        abort(500, "Could not connect to the database. Please check database configuration and status.")

# --- ROUTE UTAMA: Menampilkan Daftar Studi (Termasuk Pencarian) ---
@app.route("/", methods=['GET'])
@app.route("/search_studies", methods=['GET'])
def index():
    conn = None
    cur = None
    studies = []
    error_message = None

    # Ambil parameter pencarian dari request
    patient_id = request.args.get('patient_id', '').strip()
    patient_name = request.args.get('patient_name', '').strip()
    study_date = request.args.get('study_date', '').strip()
    accession_number = request.args.get('accession_number', '').strip()

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)

        # Query untuk mendapatkan daftar studi unik
        query = """
            SELECT DISTINCT
                st.study_prkey,
                st.study_instance_uid,
                st.study_id,
                st.study_date,
                st.study_time,
                st.study_description,
                st.accession_number,
                p.patient_id,
                p.patient_name,
                st.modalities_in_study
            FROM
                study st
            JOIN
                patient p ON st.patient_prkey = p.patient_prkey
            WHERE 1=1
        """
        params = []

        if patient_id:
            query += " AND p.patient_id ILIKE %s"
            params.append(f"%{patient_id}%")
        if patient_name:
            query += " AND p.patient_name ILIKE %s"
            params.append(f"%{patient_name}%")
        if study_date:
            query += " AND st.study_date = %s" # Pastikan format tanggal cocok
            params.append(study_date)
        if accession_number:
            query += " AND st.accession_number ILIKE %s"
            params.append(f"%{accession_number}%")

        query += """
            ORDER BY
                st.study_date DESC, p.patient_name, st.study_instance_uid
        """
        
        logging.info(f"Executing study search query: {query} with params: {params}")
        cur.execute(query, params)
        studies = cur.fetchall()
        
    except Exception as e:
        logging.exception(f"Error fetching studies from database: {e}")
        error_message = f"Failed to load studies: {e}"
    finally:
        if cur: cur.close()
        if conn: conn.close()

    return render_template( 
        'index.html', 
        studies=studies, 
        error=error_message,
        patient_id=patient_id,
        patient_name=patient_name,
        study_date=study_date,
        accession_number=accession_number
    )

# --- Helper function to fetch study details ---
def _fetch_study_details(study_instance_uid):
    conn = None
    cur = None
    study_info = None
    images_in_study = []
    error_message = None

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)

        cur.execute("""
            SELECT
                st.study_instance_uid,
                st.study_id,
                st.study_date,
                st.study_time,
                st.study_description,
                st.accession_number,
                p.patient_id,
                p.patient_name,
                p.patient_birthdate,
                p.patient_sex
            FROM
                study st
            JOIN
                patient p ON st.patient_prkey = p.patient_prkey
            WHERE st.study_instance_uid = %s;
        """, (study_instance_uid,))
        study_info = cur.fetchone()

        if study_info:
            cur.execute("""
                SELECT
                    i.image_prkey,
                    i.image_path_name,
                    i.sop_instance_uid,
                    s.series_number,
                    s.series_instance_uid, -- BARU: Tambahkan ini untuk pengelompokan seri di frontend
                    s.series_description,
                    s.modality,
                    i.image_number
                FROM
                    image i
                JOIN
                    series s ON i.series_prkey = s.series_prkey
                JOIN
                    study st ON s.study_prkey = st.study_prkey
                WHERE st.study_instance_uid = %s
                ORDER BY s.series_number, s.series_instance_uid, i.image_number; -- BARU: Pastikan urutan gambar dalam seri benar
            """, (study_instance_uid,))
            images_in_study = cur.fetchall()
        else:
            error_message = f"Study with UID '{study_instance_uid}' not found."

    except Exception as e:
        logging.exception(f"Error fetching study details for {study_instance_uid}: {e}")
        error_message = f"Failed to load study details: {e}"
    finally:
        if cur: cur.close()
        if conn: conn.close()
    
    # Konversi hasil DictRow ke dict biasa agar lebih mudah diakses di Jinja dan JSON
    study_info_dict = dict(study_info) if study_info else None
    images_in_study_list = [dict(img) for img in images_in_study]

    return study_info_dict, images_in_study_list, error_message


# --- ROUTE Untuk Menampilkan Gambar dalam Sebuah Studi ---
@app.route("/study/<study_instance_uid>", methods=['GET'])
def view_study(study_instance_uid):
    study_info, images_in_study, error_message = _fetch_study_details(study_instance_uid)

    if not study_info and not error_message: # Jika tidak ada studi dan tidak ada error dari fetch
        abort(404, f"Study with UID '{study_instance_uid}' not found.")
    
    return render_template( 
        'study_viewer.html', 
        study=study_info,
        images=images_in_study, # Ini sudah list of dictionaries
        error=error_message,
        initial_image_to_load=None # Tidak ada gambar awal untuk rute ini
    )

# --- ROUTE untuk Integrasi EMR (Lebih Fleksibel dan Menuju Fullscreen) ---
@app.route("/studyid", methods=['GET'])
def studyid():
    patient_id = request.args.get('patient_id', '').strip()
    accession_number = request.args.get('accession_number', '').strip()
    # Opsional: parameter untuk memilih gambar spesifik di dalam studi
    image_sop_instance_uid = request.args.get('image_sop_instance_uid', '').strip()

    # Periksa apakah setidaknya satu parameter identifikasi studi disediakan
    if not patient_id and not accession_number:
        flash("Patient ID or Accession Number (or both) are required for EMR integration link.", "error")
        return redirect(url_for('index'))

    conn = None
    cur = None
    study_instance_uid_found = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)

        query = """
            SELECT st.study_instance_uid
            FROM study st
            JOIN patient p ON st.patient_prkey = p.patient_prkey
            WHERE 1=1
        """
        params = []

        if patient_id:
            query += " AND p.patient_id = %s"
            params.append(patient_id)
        if accession_number:
            query += " AND st.accession_number = %s"
            params.append(accession_number)
        
        query += " ORDER BY st.study_date DESC, p.patient_name LIMIT 2;"

        logging.info(f"Executing EMR link query: {query} with params: {params}")
        cur.execute(query, params)
        results = cur.fetchall()

        if not results:
            flash(f"No study found for the provided criteria. Patient ID: '{patient_id}', Accession Number: '{accession_number}'.", "error")
            return redirect(url_for('index', patient_id=patient_id, accession_number=accession_number))
        elif len(results) > 1:
            flash(f"Multiple studies found for the provided criteria. Please refine your search. Patient ID: '{patient_id}', Accession Number: '{accession_number}'.", "warning")
            return redirect(url_for('index', patient_id=patient_id, accession_number=accession_number))
        else:
            study_instance_uid_found = results[0]['study_instance_uid']

    except Exception as e:
        logging.exception(f"Error processing EMR integration link: {e}")
        flash(f"An error occurred while trying to link to the study: {e}", "error")
        return redirect(url_for('index'))
    finally:
        if cur: cur.close()
        if conn: conn.close()

    # Jika studi ditemukan, ambil detail studi dan gambar
    if study_instance_uid_found:
        study_info, images_in_study, error_message = _fetch_study_details(study_instance_uid_found)

        initial_image_path = None
        if image_sop_instance_uid:
            # Cari gambar spesifik berdasarkan SOP Instance UID
            for img in images_in_study:
                if img['sop_instance_uid'] == image_sop_instance_uid:
                    initial_image_path = img['image_path_name']
                    break
            if not initial_image_path:
                flash(f"Image with SOP Instance UID '{image_sop_instance_uid}' not found in study.", "warning")
                logging.warning(f"Image with SOP Instance UID '{image_sop_instance_uid}' not found in study '{study_instance_uid_found}'.")
        
        # Jika tidak ada gambar spesifik diminta atau tidak ditemukan, ambil gambar pertama
        if not initial_image_path and images_in_study:
            initial_image_path = images_in_study[0]['image_path_name']
            logging.info(f"Menggunakan gambar pertama untuk studi {study_instance_uid_found}: {initial_image_path}")

        return render_template(
            'study_viewer.html',
            study=study_info,
            images=images_in_study,
            error=error_message,
            initial_image_to_load=initial_image_path # Teruskan jalur gambar awal ke templat
        )
    else:
        # Ini seharusnya tidak tercapai jika logika di atas sudah menemukan studi atau redirect
        flash("Study not found.", "error")
        return redirect(url_for('index'))

def resolve_dicom_path_for_viewer(db_path):
    """
    Mengonversi jalur file dari database (bisa absolut atau relatif)
    menjadi jalur absolut yang dapat dibaca oleh sistem file.
    """
    # Gunakan realpath untuk menormalisasi jika db_path mengandung '..' atau symlink
    # dan untuk mendapatkan jalur kanonik.
    # Jika db_path sudah absolut, os.path.join akan mengabaikan RECEIVED_DICOM_ROOT
    # jika db_path berada di drive atau root yang berbeda.
    # Maka, lebih aman memeriksa apakah db_path sudah absolut terlebih dahulu.
    
    if os.path.isabs(db_path):
        # Jika jalur dari DB sudah absolut, gunakan langsung
        # Namun, kita harus memastikan itu masih berada di dalam area yang diizinkan (RECEIVED_DICOM_ROOT)
        # Jika tidak, ini adalah kasus 'directory traversal' dari data lama yang tidak valid.
        return os.path.realpath(db_path)
    else:
        # Jika jalur dari DB relatif, gabungkan dengan RECEIVED_DICOM_ROOT
        return os.path.realpath(os.path.join(RECEIVED_DICOM_ROOT, db_path))

# --- ROUTE Untuk Mengonversi dan Mengirim Gambar PNG (dengan Windowing & Enhancement) ---
@app.route("/image_data/<path:filename_from_db>") # Menerima jalur (bisa absolut atau relatif) dari frontend
def image_data(filename_from_db):
    """
    Mengonversi gambar DICOM ke PNG dan mengirimkannya sebagai respons,
    mengizinkan parameter Window Level (wl), Window Width (ww), dan enhancement opsional.
    """
    # Resolusi jalur file dari database menjadi jalur absolut sistem file
    full_dicom_path = resolve_dicom_path_for_viewer(filename_from_db)

    # Validasi keamanan: Pastikan jalur yang telah diselesaikan berada di dalam RECEIVED_DICOM_ROOT
    if not full_dicom_path.startswith(RECEIVED_DICOM_ROOT):
        logging.error(f"Percobaan directory traversal terdeteksi di image_data: {filename_from_db} (resolved to {full_dicom_path})")
        abort(403, "Terlarang: Jalur file tidak valid.")

    if not os.path.exists(full_dicom_path):
        logging.error(f"File DICOM '{full_dicom_path}' tidak ditemukan untuk konversi (diminta: {filename_from_db}).")
        abort(404, f"File DICOM '{filename_from_db}' tidak ditemukan untuk konversi.")

    try:
        ds = dcmread(full_dicom_path, force=True) 

        if 'PixelData' not in ds:
            logging.error(f"Tidak ada data piksel di file DICOM: {filename_from_db}")
            abort(400, "File DICOM tidak mengandung data piksel.")

        pixel_array_raw = ds.pixel_array # Dapatkan raw pixel array

        # --- PERBAIKAN DI SINI: Tangani gambar multi-frame dan PALETTE COLOR ---
        pixel_array_orig = None
        
        # Periksa apakah ini gambar multi-frame (jika NumberOfFrames > 1)
        num_frames = ds.get('NumberOfFrames', 1)
        if num_frames > 1 and pixel_array_raw.ndim > 2:
            # Jika multi-frame (misalnya, (frames, rows, cols) atau (frames, rows, cols, channels))
            # Ambil frame pertama secara default untuk tampilan thumbnail
            pixel_array_orig = pixel_array_raw[0].astype(np.float32)
            logging.info(f"Gambar multi-frame terdeteksi untuk {filename_from_db}, memproses frame pertama.")
        else:
            # Gambar single frame
            pixel_array_orig = pixel_array_raw.astype(np.float32)
        # --- AKHIR PERBAIKAN ---

        # Menangani Rescale Slope dan Intercept (sekarang diterapkan pada pixel_array_orig yang sudah single frame)
        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
            rescale_slope = float(ds.RescaleSlope)
            rescale_intercept = float(ds.RescaleIntercept)
            pixel_array_orig = pixel_array_orig * rescale_slope + rescale_intercept
        
        pixel_array_processed = pixel_array_orig.copy()

        # Dapatkan Window Level dan Window Width dari parameter query atau metadata DICOM
        default_wl = ds.get("WindowLevel", None) 
        default_ww = ds.get("WindowWidth", None)
        
        if isinstance(default_wl, multival.MultiValue):
            default_wl = default_wl[0]
        if isinstance(default_ww, multival.MultiValue):
            default_ww = default_ww[0]
            
        wl = float(request.args.get("wl", default_wl)) if request.args.get("wl") else default_wl
        ww = float(request.args.get("ww", default_ww)) if request.args.get("ww") else default_ww

        # Jika WL atau WW masih None (tidak ada di DICOM dan tidak di URL), hitung dari min/max pixel
        if wl is None or ww is None:
            min_pixel = np.min(pixel_array_processed)
            max_pixel = np.max(pixel_array_processed)
            wl = (min_pixel + max_pixel) / 2.0
            ww = max_pixel - min_pixel if (max_pixel - min_pixel) > 0 else 1.0 

        # Terapkan Windowing
        if ww > 0: 
            min_val = wl - ww / 2.0 
            max_val = wl + ww / 2.0 
            pixel_array_processed = np.clip(pixel_array_processed, min_val, max_val)
            pixel_array_processed = ((pixel_array_processed - min_val) / (max_val - min_val)) * 255.0
        else:
            # Jika WW nol, gambar akan menjadi biner (hitam atau putih)
            pixel_array_processed = np.where(pixel_array_processed >= wl, 255.0, 0.0)

        pixel_array_processed = np.clip(pixel_array_processed, 0, 255).astype(np.uint8)

        # --- Image Enhancement (CLAHE) ---
        enhance_mode = request.args.get("enhance", "").lower()
        if enhance_mode == "clahe":
            logging.info(f"Menerapkan CLAHE untuk {filename_from_db}")
            img_as_float = pixel_array_processed.astype(np.float32) / 255.0
            pixel_array_enhanced = exposure.equalize_adapthist(img_as_float, 
                                                                clip_limit=0.03, 
                                                                kernel_size=(8, 8)) 
            pixel_array_processed = (pixel_array_enhanced * 255).astype(np.uint8)
        # --- End Image Enhancement ---

        img = None
        photometric_interpretation = ds.get("PhotometricInterpretation", "").upper()

        # Penanganan interpretasi fotometrik
        # Pada titik ini, pixel_array_processed seharusnya sudah satu frame.
        if photometric_interpretation == "MONOCHROME1":
            img = Image.fromarray(255 - pixel_array_processed) # Invert for MONOCHROME1
        elif photometric_interpretation == "MONOCHROME2" or photometric_interpretation == "":
            img = Image.fromarray(pixel_array_processed)
        elif photometric_interpretation == "RGB":
            if pixel_array_processed.ndim == 3 and pixel_array_processed.shape[-1] == 3:
                img = Image.fromarray(pixel_array_processed)
            else:
                logging.error(f"Bentuk data piksel RGB tidak didukung untuk {filename_from_db}: {pixel_array_processed.shape}")
                abort(400, f"Bentuk data piksel RGB tidak didukung: {pixel_array_processed.shape}")
        elif photometric_interpretation in ["YBR_FULL_422", "YBR_FULL", "YBR_ICT", "YBR_RCT"]:
            # Untuk YBR, asumsikan sudah dikonversi ke RGB jika pydicom memprosesnya
            if pixel_array_processed.ndim == 3 and pixel_array_processed.shape[-1] == 3:
                 img = Image.fromarray(pixel_array_processed)
            else:
                logging.error(f"Konversi {photometric_interpretation} gagal atau format tidak didukung untuk {filename_from_db}.")
                abort(400, f"Konversi {photometric_interpretation} gagal atau format tidak didukung.")
        elif photometric_interpretation == "PALETTE COLOR":
            # pydicom.pixel_array sudah mengonversi ini ke grayscale atau RGB
            # Jadi kita bisa meneruskannya seperti MONOCHROME2 atau RGB
            if pixel_array_processed.ndim == 2: # Berarti dikonversi ke grayscale
                img = Image.fromarray(pixel_array_processed)
            elif pixel_array_processed.ndim == 3 and pixel_array_processed.shape[-1] == 3: # Berarti dikonversi ke RGB
                img = Image.fromarray(pixel_array_processed)
            else:
                logging.error(f"Bentuk data piksel PALETTE COLOR yang dikonversi tidak didukung untuk {filename_from_db}: {pixel_array_processed.shape}")
                abort(400, f"Bentuk data piksel PALETTE COLOR yang dikonversi tidak didukung: {pixel_array_processed.shape}")
        else:
            # Penanganan umum untuk kasus lain, mencoba memperlakukannya sebagai citra grayscale atau RGB
            if pixel_array_processed.ndim == 2:
                img = Image.fromarray(pixel_array_processed)
            elif pixel_array_processed.ndim == 3 and pixel_array_processed.shape[-1] in [1, 3]: 
                # Jika 1 channel, ambil channel pertama; jika 3, gunakan apa adanya
                img = Image.fromarray(pixel_array_processed[:,:,0] if pixel_array_processed.shape[-1] == 1 else pixel_array_processed)
            else:
                logging.error(f"Interpretasi Fotometrik '{photometric_interpretation}' tidak didukung atau bentuk gambar untuk {filename_from_db}: {pixel_array_processed.shape}")
                abort(400, f"Interpretasi Fotometrik '{photometric_interpretation}' tidak didukung atau bentuk gambar: {pixel_array_processed.shape}")

        if img is None:
             logging.error(f"Tidak dapat membuat gambar dari data piksel DICOM untuk {filename_from_db}.")
             abort(400, "Tidak dapat membuat gambar dari data piksel DICOM.")

        img_io = BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)

        response = send_file(img_io, mimetype='image/png')
        response.headers['X-Current-WL'] = str(wl) 
        response.headers['X-Current-WW'] = str(ww)
        response.headers['X-Current-Enhance'] = enhance_mode # Kirim status enhancement
        return response

    except pydicom.errors.InvalidDicomError as e:
        logging.error(f"File DICOM tidak valid '{filename_from_db}': {e}")
        abort(400, f"File DICOM tidak valid. {e}")
    except Exception as e:
        logging.exception(f"Kesalahan saat menampilkan gambar '{filename_from_db}': {e}")
        abort(500, f"Kesalahan saat memproses gambar DICOM: {e}")

# --- ROUTE Untuk Mendapatkan Metadata Default Windowing (untuk inisialisasi popup) ---
@app.route("/image_metadata/<path:filename_from_db>") 
def image_metadata(filename_from_db):
    """
    Mengembalikan default Window Level dan Window Width dari file DICOM.
    Digunakan oleh frontend untuk inisialisasi kontrol windowing.
    """
    full_dicom_path = resolve_dicom_path_for_viewer(filename_from_db)

    # Validasi keamanan
    if not full_dicom_path.startswith(RECEIVED_DICOM_ROOT):
        logging.error(f"Percobaan directory traversal terdeteksi di metadata request: {filename_from_db} (resolved to {full_dicom_path})")
        abort(403, "Terlarang: Jalur file tidak valid.")

    if not os.path.exists(full_dicom_path):
        abort(404, f"File DICOM '{filename_from_db}' tidak ditemukan.")
    
    try:
        ds = dcmread(full_dicom_path, force=True)
        
        # --- PERBAIKAN DI SINI: Tangani multi-frame untuk metadata juga ---
        pixel_array_raw = ds.pixel_array
        if ds.get('NumberOfFrames', 1) > 1 and pixel_array_raw.ndim > 2:
            pixel_array_for_minmax = pixel_array_raw[0] # Ambil frame pertama untuk perhitungan min/max
            logging.info(f"Multi-frame image detected for metadata {filename_from_db}, calculating min/max from first frame.")
        else:
            pixel_array_for_minmax = pixel_array_raw
        # --- AKHIR PERBAIKAN ---

        default_wl = ds.get("WindowLevel", None) 
        default_ww = ds.get("WindowWidth", None)

        if isinstance(default_wl, multival.MultiValue):
            default_wl = default_wl[0]
        if isinstance(default_ww, multival.MultiValue):
            default_ww = default_ww[0]
        
        if default_wl is None or default_ww is None:
            if 'PixelData' in ds:
                # Gunakan pixel_array_for_minmax yang sudah dipastikan satu frame
                pixel_array = pixel_array_for_minmax 
                if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
                    rescale_slope = float(ds.RescaleSlope)
                    rescale_intercept = float(ds.RescaleIntercept)
                    pixel_array = pixel_array * rescale_slope + rescale_intercept
                
                min_pixel = np.min(pixel_array)
                max_pixel = np.max(pixel_array)
                default_wl = (min_pixel + max_pixel) / 2.0
                default_ww = max_pixel - min_pixel if (max_pixel - min_pixel) > 0 else 1.0
            else:
                default_wl = 127.0
                default_ww = 255.0
        
        # --- PERBAIKAN DI SINI: Konversi ke float untuk JSON serialization ---
        # Pastikan default_wl dan default_ww adalah float standar Python
        return {
            "default_wl": float(default_wl),
            "default_ww": float(default_ww),
            "filename": filename_from_db # Kembalikan nama file seperti yang diterima
        }
    except Exception as e:
        logging.error(f"Kesalahan saat mendapatkan metadata untuk '{filename_from_db}': {e}")
        abort(500, f"Kesalahan saat mendapatkan metadata: {e}")


if __name__ == "__main__":
    app.run(debug=True, host=app_host, port=app_port)