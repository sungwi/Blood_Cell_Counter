import os

# Paths
folder = "./dataset_2/raw/platelet"

# Grab files
original_files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg') and '_processed' not in f])

# Mapping from original name to newer index-based name
mapping = {}
for idx, fname in enumerate(original_files, 1):
    base, ext = os.path.splitext(fname)
    new_name = f"Platelet_{idx}{ext}"
    mapping[base] = new_name

# Apply Renaming
for old_base, new_name in mapping.items():
    original_path = os.path.join(folder, f"{old_base}.jpg")

    # Rename original
    if os.path.exists(original_path):
        os.rename(original_path, os.path.join(folder, new_name))

print("Done.")
