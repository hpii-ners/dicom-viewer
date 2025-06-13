import pydicom
from pydicom.errors import InvalidDicomError

dicom_file = "1.2.840.113704.7.1.0.47929410081112.1745247885.15.dcm"
output_file = "metadata.txt"

try:
    # Coba baca file secara normal
    ds = pydicom.dcmread(dicom_file)
except InvalidDicomError as e:
    print(f"[INFO] File DICOM tidak valid secara standar: {e}")
    print("[INFO] Mencoba baca dengan force=True...")
    try:
        ds = pydicom.dcmread(dicom_file, force=True)
    except Exception as ex:
        print(f"[ERROR] Gagal membaca file bahkan dengan force=True: {ex}")
        exit(1)

# Tulis metadata ke file teks
with open(output_file, "w") as f:
    for elem in ds:
        line = f"{elem.tag} : {elem.keyword} = {elem.value}"
        print(line)        # tampilkan ke console
        f.write(line + "\n")

print(f"[SUCCESS] Metadata berhasil disimpan ke '{output_file}'")
