import os

# folder = "./dataset_2/raw/basophil"

folder = "./dataset_2/processed/basophil_processed"

files = sorted([f for f in os.listdir(folder) if f.startswith("Basophil") and f.endswith(".jpg")])

for idx, filename in enumerate(files, 1):
    ext = os.path.splitext(filename)[1]
    new_name = f"Basophil_{idx:04d}{ext}"  # Pads with 0 to 4 digits
    os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))

print("Renamed with 0 padding.")
