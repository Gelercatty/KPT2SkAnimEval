import os
import numpy as np

dir_52_point = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\52point"

dir_taichi = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\taichi"

file_list = os.listdir(dir_52_point)

for item in file_list:
    data_52_item_path     = os.path.join(dir_52_point, item)
    data_taichi_item_path = os.path.join(dir_taichi, item.split('.')[0]+'.npy')
    
    data_52_item = np.load(data_52_item_path)
    data_taichi_item = data_52_item[:, :22, :].copy()
    np.save(data_taichi_item_path, data_taichi_item)
