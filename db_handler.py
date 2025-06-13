# db_handler.py
import psycopg2
import pydicom 
from pydicom import dcmread, multival 
import configparser
import logging
import os

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Dapatkan root direktori proyek (dari tempat db_handler.py berada)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Definisikan RECEIVED_DICOM_ROOT secara konsisten
RECEIVED_DICOM_ROOT = os.path.realpath(os.path.join(BASE_DIR, 'received_dicom'))

# Fungsi helper untuk nilai default dan penanganan MultiValue
def get_str(value):
    """
    Mengonversi nilai DICOM (termasuk MultiValue) ke string,
    mengembalikan string kosong jika nilai None atau kosong.
    """
    if value is None:
        return ""
    if isinstance(value, (multival.MultiValue, list, tuple)):
        # Jika MultiValue, ambil elemen pertama atau kembalikan string kosong jika kosong
        return str(value[0]).strip() if value else ""
    return str(value).strip()

def get_int(value):
    """
    Mengonversi nilai DICOM (termasuk MultiValue) ke integer,
    mengembalikan 0 jika nilai None atau tidak dapat dikonversi.
    """
    try:
        if value is None:
            return 0
        if isinstance(value, (multival.MultiValue, list, tuple)):
            return int(value[0]) if value else 0
        return int(value)
    except (ValueError, TypeError):
        return 0

# Load konfigurasi
config = configparser.ConfigParser()
try:
    config.read('config.ini')
    if 'database' not in config:
        raise ValueError("Bagian '[database]' tidak ditemukan di config.ini")
except Exception as e:
    logging.error(f"Gagal membaca config.ini atau bagian '[database]' tidak ada: {e}")
    logging.error("Pastikan Anda memiliki file config.ini dengan bagian [database] yang benar.")
    exit(1) # Keluar jika konfigurasi database tidak dapat dimuat

# Inisialisasi koneksi database secara global
conn = None
try:
    db_config = config['database']
    conn = psycopg2.connect(
        host=db_config.get('host', 'localhost'),
        port=db_config.getint('port', 5432),
        database=db_config.get('database'),
        user=db_config.get('user'),
        password=db_config.get('password')
    )
    conn.autocommit = True
    logging.info("Koneksi database berhasil dibuat.")
except psycopg2.Error as e:
    logging.error(f"Gagal terhubung ke database: {e}")
    logging.error("Pastikan server PostgreSQL berjalan dan kredensialnya benar di config.ini.")
    exit(1) # Keluar jika koneksi database gagal

def get_or_create_patient(ds, cur, file_path_relative): 
    """
    Mendapatkan patient_prkey dari PatientID yang ada,
    atau membuat entri pasien baru jika belum ada.
    """
    patient_id = get_str(ds.get("PatientID", None))
    if not patient_id: # Pastikan PatientID tidak kosong
        patient_id = "UNKNOWN_PATIENT_" + str(ds.SOPInstanceUID) # Fallback jika PatientID kosong
        logging.warning(f"PatientID kosong, menggunakan fallback: {patient_id}")

    cur.execute("SELECT patient_prkey FROM public.patient WHERE patient_id = %s;", (patient_id,))
    result = cur.fetchone()

    if result:
        logging.debug(f"Pasien dengan PatientID '{patient_id}' sudah ada, menggunakan prkey: {result[0]}")
        return result[0]
    else:
        logging.debug(f"Membuat pasien baru dengan PatientID: {patient_id}")
        patient_name = get_str(ds.get("PatientName", None))
        patient_birthdate = get_str(ds.get("PatientBirthDate", None))
        patient_sex = get_str(ds.get("PatientSex", None))
        character_set = get_str(ds.get("SpecificCharacterSet", None))
        patient_path = get_str(file_path_relative) # Gunakan jalur relatif yang diteruskan

        cur.execute("""
            INSERT INTO public.patient (patient_id, patient_name, patient_birthdate, patient_sex, character_set, patient_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING patient_prkey;
        """, (patient_id, patient_name, patient_birthdate, patient_sex, character_set, patient_path))
        return cur.fetchone()[0]

def get_or_create_study(ds, cur, patient_prkey, file_path_relative): # Gunakan jalur relatif
    """
    Mendapatkan study_prkey dari StudyInstanceUID yang ada,
    atau membuat entri studi baru jika belum ada.
    """
    study_instance_uid = get_str(ds.get("StudyInstanceUID", None))
    if not study_instance_uid:
        study_instance_uid = "UNKNOWN_STUDY_" + str(ds.SOPInstanceUID)
        logging.warning(f"StudyInstanceUID kosong, menggunakan fallback: {study_instance_uid}")

    cur.execute("SELECT study_prkey FROM public.study WHERE study_instance_uid = %s;", (study_instance_uid,))
    result = cur.fetchone()

    if result:
        logging.debug(f"Studi dengan StudyInstanceUID '{study_instance_uid}' sudah ada, menggunakan prkey: {result[0]}")
        return result[0]
    else:
        logging.debug(f"Membuat studi baru dengan StudyInstanceUID: {study_instance_uid}")
        # Ambil data studi
        study_id = get_str(ds.get("StudyID", None))
        study_date = get_str(ds.get("StudyDate", None))
        study_time = get_str(ds.get("StudyTime", None))
        study_description = get_str(ds.get("StudyDescription", None))
        
        # --- PERBAIKAN DI SINI: Fallback untuk AccessionNumber ---
        accession_number = get_str(ds.get("AccessionNumber", None))
        if not accession_number: # Jika AccessionNumber kosong, gunakan StudyInstanceUID sebagai fallback
            accession_number = study_instance_uid
            logging.warning(f"AccessionNumber kosong, menggunakan StudyInstanceUID '{study_instance_uid}' sebagai fallback.")
        # --- AKHIR PERBAIKAN ---

        refer_physician = get_str(ds.get("ReferringPhysicianName", None))
        modalities_in_study = get_str(ds.get("ModalitiesInStudy", None))
        series_in_study = get_str(ds.get("NumberOfStudyRelatedSeries", None))
        images_in_study = get_str(ds.get("NumberOfStudyRelatedInstances", None))
        station_name = get_str(ds.get("StationName", None))
        institutional_department_name = get_str(ds.get("InstitutionalDepartmentName", None))
        patient_age = get_str(ds.get("PatientAge", None))
        patient_weight = get_str(ds.get("PatientWeight", None))
        institution_name = get_str(ds.get("InstitutionName", None))
        frames_in_study = get_str(ds.get("NumberOfFrames", None)) 
        study_comments = get_str(ds.get("StudyComments", None))
        character_set = get_str(ds.get("SpecificCharacterSet", None))
        study_path = get_str(file_path_relative) # Gunakan jalur relatif yang diteruskan

        cur.execute("""
            INSERT INTO public.study (
                study_instance_uid, study_id, study_date, study_time, study_description, accession_number,
                refer_physician, modalities_in_study, series_in_study, images_in_study, station_name,
                institutional_department_name, patient_age, patient_weight, character_set, study_path,
                patient_prkey, institution_name, frames_in_study, study_comments, study_status, study_token
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING study_prkey;
        """, (
            study_instance_uid, study_id, study_date, study_time, study_description, accession_number, # accession_number yang sudah difallback
            refer_physician, modalities_in_study, series_in_study, images_in_study, station_name,
            institutional_department_name, patient_age, patient_weight, character_set, study_path,
            patient_prkey, institution_name, frames_in_study, study_comments, 0, "none"
        ))
        return cur.fetchone()[0]

def get_or_create_series(ds, cur, study_prkey, file_path_relative): # Gunakan jalur relatif
    """
    Mendapatkan series_prkey dari SeriesInstanceUID yang ada,
    atau membuat entri series baru jika belum ada.
    """
    series_instance_uid = get_str(ds.get("SeriesInstanceUID", None))
    if not series_instance_uid:
        series_instance_uid = "UNKNOWN_SERIES_" + str(ds.SOPInstanceUID)
        logging.warning(f"SeriesInstanceUID kosong, menggunakan fallback: {series_instance_uid}")

    cur.execute("SELECT series_prkey FROM public.series WHERE series_instance_uid = %s;", (series_instance_uid,))
    result = cur.fetchone()

    if result:
        logging.debug(f"Series dengan SeriesInstanceUID '{series_instance_uid}' sudah ada, menggunakan prkey: {result[0]}")
        return result[0]
    else:
        logging.debug(f"Membuat series baru dengan SeriesInstanceUID: {series_instance_uid}")
        # Ambil data series
        series_number = get_str(ds.get("SeriesNumber", None))
        series_date = get_str(ds.get("SeriesDate", None))
        series_time = get_str(ds.get("SeriesTime", None))
        series_description = get_str(ds.get("SeriesDescription", None))
        modality = get_str(ds.get("Modality", None))
        patient_position = get_str(ds.get("PatientPosition", None))
        contrast_bolus_agent = get_str(ds.get("ContrastBolusAgent", None))
        manufacturer = get_str(ds.get("Manufacturer", None))
        model_name = get_str(ds.get("ManufacturerModelName", None))
        body_part_examined = get_str(ds.get("BodyPartExamined", None))
        protocol_name = get_str(ds.get("ProtocolName", None))
        images_in_series = get_str(ds.get("NumberOfSeriesRelatedInstances", None))
        frame_of_reference_uid = get_str(ds.get("FrameOfReferenceUID", None))
        localizer_instance_uid = get_str(ds.get("SOPInstanceUID", None))  # SOPInstanceUID dari image ini
        user_comments = "none"
        series_priority = 0
        series_rate = 0
        series_review_date = "none"
        series_review_time = "none"
        series_insertion_date = "none"
        character_set = get_str(ds.get("SpecificCharacterSet", None))
        study_instance_uid_ref = get_str(ds.get("StudyInstanceUID", None)) # Referensi StudyInstanceUID
        frames_in_series = get_str(ds.get("NumberOfFrames", None)) # Ini juga bisa jadi milik Image
        series_path = get_str(file_path_relative) # Gunakan jalur relatif yang diteruskan

        cur.execute("""
            INSERT INTO public.series (
                series_instance_uid, series_number, series_date, series_time, series_description, modality,
                patient_position, contrast_bolus_agent, manufacturer, model_name, body_part_examined,
                protocol_name, images_in_series, frame_of_reference_uid, localizer_instance_uid, user_comments,
                series_priority, series_rate, series_review_date, series_review_time, series_insertion_date,
                character_set, series_path, study_instance_uid, study_prkey, frames_in_series
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING series_prkey;
        """, (
            series_instance_uid, series_number, series_date, series_time, series_description, modality,
            patient_position, contrast_bolus_agent, manufacturer, model_name, body_part_examined,
            protocol_name, images_in_series, frame_of_reference_uid, localizer_instance_uid, user_comments,
            series_priority, series_rate, series_review_date, series_review_time, series_insertion_date,
            character_set, series_path, study_instance_uid_ref, study_prkey, frames_in_series
        ))
        return cur.fetchone()[0]

def save_image_metadata(ds, cur, patient_prkey, series_prkey):
    """
    Menyimpan metadata gambar. Diasumsikan SOPInstanceUID unik untuk setiap gambar,
    sehingga tidak ada pengecekan duplikat untuk image_prkey.
    """
    sop_instance_uid = get_str(ds.get("SOPInstanceUID", None))
    if not sop_instance_uid:
        logging.error("SOPInstanceUID kosong, tidak bisa menyimpan image metadata.")
        return # Tidak bisa menyimpan jika SOPInstanceUID tidak ada

    # Cek apakah image dengan SOPInstanceUID ini sudah ada untuk menghindari duplikat
    cur.execute("SELECT image_prkey FROM public.image WHERE sop_instance_uid = %s;", (sop_instance_uid,))
    result = cur.fetchone()

    if result:
        logging.info(f"Image dengan SOPInstanceUID '{sop_instance_uid}' sudah ada, tidak ada penyisipan baru.")
        return # Jangan sisipkan jika sudah ada
    
    logging.debug(f"Membuat image baru dengan SOPInstanceUID: {sop_instance_uid}")
    sop_class_uid = get_str(ds.get("SOPClassUID", None))
    image_type = get_str(ds.get("ImageType", None))
    image_number = get_str(ds.get("InstanceNumber", None))
    image_date = get_str(ds.get("ContentDate", None))
    image_time = get_str(ds.get("ContentTime", None))
    acquisition_date = get_str(ds.get("AcquisitionDate", None))
    acquisition_time = get_str(ds.get("AcquisitionTime", None))
    acquisition_number = get_str(ds.get("AcquisitionNumber", None))
    number_of_frames = get_str(ds.get("NumberOfFrames", None))
    samples_per_pixel = get_str(ds.get("SamplesPerPixel", None))
    photometric_interpretation = get_str(ds.get("PhotometricInterpretation", None))
    bits_allocated = get_int(ds.get("BitsAllocated", None))
    rows = get_int(ds.get("Rows", None))
    columns = get_int(ds.get("Columns", None))
    # 'Path' sekarang diharapkan berisi jalur relatif dari server DICOM
    image_path_name = get_str(ds.get("Path", None)) 
    patient_id = get_str(ds.get("PatientID", None)) 

    cur.execute("""
        INSERT INTO public.image (
            sop_instance_uid, sop_class_uid, image_type, image_number, image_date, image_time,
            acquisition_date, acquisition_time, acquisition_number, number_of_frames,
            samples_per_pixel, photometric_interpretation, bits_allocated, rows, columns,
            image_path_name, patient_id, series_prkey
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        sop_instance_uid, sop_class_uid, image_type, image_number, image_date, image_time,
        acquisition_date, acquisition_time, acquisition_number, number_of_frames,
        samples_per_pixel, photometric_interpretation, bits_allocated, rows, columns,
        image_path_name, patient_id, series_prkey
    ))


def save_metadata(path_relative_to_received_dir): # Menerima jalur relatif dari RECEIVED_DIR
    """
    Fungsi utama untuk membaca file DICOM dan menyimpan metadatanya ke database.
    """
    try:
        # Untuk membaca file DICOM, kita memerlukan jalur absolutnya
        # Gabungkan dengan RECEIVED_DICOM_ROOT yang sudah didefinisikan di db_handler
        filepath_absolute = os.path.join(RECEIVED_DICOM_ROOT, path_relative_to_received_dir)
        
        # Periksa apakah file benar-benar ada sebelum mencoba membacanya
        if not os.path.exists(filepath_absolute):
            logging.error(f"File DICOM tidak ditemukan di jalur absolut: {filepath_absolute}")
            raise FileNotFoundError(f"File DICOM tidak ditemukan: {filepath_absolute}")

        ds = dcmread(filepath_absolute)
        
        # Tambahkan jalur RELATIF ke dataset agar bisa diakses oleh fungsi save_image_metadata
        ds.Path = path_relative_to_received_dir 
        cur = conn.cursor()

        patient_prkey = get_or_create_patient(ds, cur, path_relative_to_received_dir) 
        study_prkey = get_or_create_study(ds, cur, patient_prkey, path_relative_to_received_dir)
        series_prkey = get_or_create_series(ds, cur, study_prkey, path_relative_to_received_dir)
        save_image_metadata(ds, cur, patient_prkey, series_prkey)

        logging.info(f"Metadata dari '{path_relative_to_received_dir}' berhasil disimpan ke database.")
    except pydicom.errors.InvalidDicomError as e:
        logging.error(f"File DICOM tidak valid '{path_relative_to_received_dir}': {e}")
        raise # Re-raise exception agar server bisa menangani
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat menyimpan metadata untuk '{path_relative_to_received_dir}': {e}")
        raise # Re-raise exception
    finally:
        if cur: cur.close()

# Contoh penggunaan (opsional, untuk testing)
if __name__ == '__main__':
    logging.info("db_handler.py dijalankan langsung. Ini biasanya digunakan untuk pengujian.")
    logging.info("Pastikan ada file DICOM yang bisa dibaca dan database yang sudah diatur.")
    # Untuk menjalankan ini, Anda perlu memiliki file DICOM yang valid.
    # Contoh:
    # from pydicom.data import get_testdata_files
    # try:
    #     # Ganti dengan path ke file DICOM yang ada di sistem Anda
    #     # Pastikan path ini RELATIF terhadap direktori 'received_dicom'
    #     test_dicom_relative_path = os.path.join('1059953627', '1.2.840.113704.7.1.0.47929410081112.1745247885.15.dcm')
    #     if os.path.exists(os.path.join(RECEIVED_DICOM_ROOT, test_dicom_relative_path)):
    #         save_metadata(test_dicom_relative_path)
    #         logging.info(f"Test metadata saved for {test_dicom_relative_path}")
    #     else:
    #         logging.warning(f"File uji tidak ditemukan: {os.path.join(RECEIVED_DICOM_ROOT, test_dicom_relative_path)}. Silakan letakkan file DICOM di sana untuk menguji.")
    # except Exception as e:
    #     logging.error(f"Error during test: {e}")

