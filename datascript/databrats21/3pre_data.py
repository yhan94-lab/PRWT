
import os
import numpy as np

mask_folder_path =r"data\npyMask"
output_file_path = r'txt\cls_labels_onehot.npy'
all_one_hot_encoded = {}

for mask_file in os.listdir(mask_folder_path):

    if mask_file.endswith('.npy'):
        mask_path = os.path.join(mask_folder_path, mask_file)
        mask = np.load(mask_path)
        unique_classes = np.unique(mask)
        num_classes = 3

        one_hot_encoded = np.zeros((num_classes), dtype=np.uint8)

        for class_index in unique_classes:
            if class_index == 1:
                one_hot_encoded[0] = 1
            if class_index == 2:
                one_hot_encoded[1] = 1
            if class_index == 4:
                one_hot_encoded[2] = 1

        file_key = os.path.splitext(mask_file)[0]
        all_one_hot_encoded[file_key] = one_hot_encoded

np.save(output_file_path, all_one_hot_encoded)

print(f"One-Hot")