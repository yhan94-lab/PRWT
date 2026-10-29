import os
import shutil
import random
import os
import numpy as np
import SimpleITK as sitk

flair_name = "-t2f.nii.gz"
t1_name = "-t1n.nii.gz"
t1ce_name = "-t1c.nii.gz"
t2_name = "-t2w.nii.gz"
mask_name = "-seg.nii.gz"


data_dir = 'BraTS2023-MEN-Train'
output_train_file = r'txt\rtrain.txt'
output_val_file = r'txt\rval.txt'
output_test_file = r'txt\rtest.txt'
outputImg_path = r"datamen\npyImage"
outputMask_path = r"datamen\npyMask"



if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)
def file_name_path(file_dir, dir=False, file=True):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """

    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return np.zeros_like(slice)
    else:
        norm_vol = (slice - slice.min()) / (slice.max() - slice.min())
        return norm_vol

def crop_ceter(img, croph, cropw):
    # for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]

def proer(pathhgg_list,f):

    for subsetindex in range(0, len(pathhgg_list),1):

        brats_subset_path = data_dir + "/" + str(pathhgg_list[subsetindex]) + "/"

        flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_name
        mask_image = brats_subset_path + str(pathhgg_list[subsetindex]) + mask_name

        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)

        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        mask_array = sitk.GetArrayFromImage(mask)

        flair_array_nor = normalize(flair_array)
        t1_array_nor = normalize(t1_array)
        t1ce_array_nor = normalize(t1ce_array)
        t2_array_nor = normalize(t2_array)

        flair_crop = crop_ceter(flair_array_nor, 160, 160)
        t1_crop = crop_ceter(t1_array_nor, 160, 160)
        t1ce_crop = crop_ceter(t1ce_array_nor, 160, 160)
        t2_crop = crop_ceter(t2_array_nor, 160, 160)
        mask_crop = crop_ceter(mask_array, 160, 160)

        numbers = [2, 3, 4, 5]
        random_number = random.choice(numbers)
        for n_slice in range(0, flair_crop.shape[0], random_number):

            flair_slice = flair_crop[n_slice, :, :]
            t1_slice = t1_crop[n_slice, :, :]
            t1ce_slice = t1ce_crop[n_slice, :, :]
            t2_slice = t2_crop[n_slice, :, :]
            flair_nonzero = flair_slice[np.nonzero(flair_slice)]
            t1_nonzero = t1_slice[np.nonzero(t1_slice)]
            t1ce_nonzero = t1ce_slice[np.nonzero(t1ce_slice)]
            t2_nonzero = t2_slice[np.nonzero(t2_slice)]
            if ((np.std(mask_crop[n_slice, :, :]) != 0) and (np.std(flair_slice) != 0 and np.std(flair_nonzero) != 0)
                    and (np.std(t1_slice) != 0 and np.std(t1_nonzero) != 0)
                    and (np.std(t1ce_slice) != 0 and np.std(t1ce_nonzero) != 0)
                    and (np.std(t2_slice) != 0 and np.std(t2_nonzero) != 0)):

                maskImg = mask_crop[n_slice, :, :]

                FourModelImageArray = np.zeros((flair_crop.shape[1], flair_crop.shape[2], 4), np.float32)
                flairImg = flair_crop[n_slice, :, :]
                flairImg = flairImg.astype(np.float32)
                FourModelImageArray[:, :, 0] = flairImg
                t1Img = t1_crop[n_slice, :, :]
                t1Img = t1Img.astype(np.float32)
                FourModelImageArray[:, :, 1] = t1Img
                t1ceImg = t1ce_crop[n_slice, :, :]
                t1ceImg = t1ceImg.astype(np.float32)
                FourModelImageArray[:, :, 2] = t1ceImg
                t2Img = t2_crop[n_slice, :, :]
                t2Img = t2Img.astype(np.float32)
                FourModelImageArray[:, :, 3] = t2Img

                if FourModelImageArray[0, 0, 0]!=0 or FourModelImageArray[0, 0, 1]!=0 or FourModelImageArray[0, 0, 2]!=0 or FourModelImageArray[0, 0, 3]!=0:
                    continue

                imagepath = outputImg_path + "\\" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
                maskpath = outputMask_path + "\\" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
                np.save(imagepath, FourModelImageArray)  # (160,160,4) np.float dtype('float64')
                np.save(maskpath, maskImg)  # (160, 160) dtype('uint8')

                outtxt_name=str(pathhgg_list[subsetindex]) + "_" + str(n_slice)
                f.write(outtxt_name + '\n')


split_ratio = 0.6


all_files = os.listdir(data_dir)

random.shuffle(all_files)
split_index = int(len(all_files) * split_ratio)

with open(output_train_file, 'w') as f:
    for file in all_files[:split_index]:

        pathhgg_list = [file]
        proer(pathhgg_list, f)

with open(output_val_file, 'w') as f:
    for file in all_files[split_index:int(split_index+(len(all_files)-split_index)/2)]:
        pathhgg_list = [file]
        proer(pathhgg_list, f)

with open(output_test_file, 'w') as f:
    for file in all_files[int(split_index+(len(all_files)-split_index)/2):]:
        pathhgg_list = [file]
        proer(pathhgg_list, f)

print("！")