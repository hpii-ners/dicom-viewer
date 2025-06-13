import monai
print("UNesT available:", hasattr(monai.networks.nets, "UNesT"))

from monai.networks.nets import UNesT
print("UNesT imported successfully!")