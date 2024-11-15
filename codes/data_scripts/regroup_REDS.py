import os
import glob

train_path = '../datasets/REDS/train_sharp'
val_path = '../datasets/REDS/train_sharp'

# mv the val set
val_folders = glob.glob(os.path.join(val_path, '*'))
for folder in val_folders:
    new_folder_idx = '{:03d}'.format(int(folder.split('/')[-1]) + 240)
    os.system('cp -r {} {}'.format(folder, os.path.join(train_path, new_folder_idx)))
