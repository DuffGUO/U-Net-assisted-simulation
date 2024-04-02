import os
from torch.utils.data import Dataset
from excel_to_matrix import *
import torch
from torchvision import transforms

class SEGData(Dataset):

    def __init__(self, input_dir, output_dir):

        # 输入input和output的相对路径，并且获取名称为列表
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_path = os.listdir(self.input_dir)

    def __len__(self):
        return len(self.input_path)

    def __getitem__(self, item):

        # 利用input和output的同样名称，完成二者在不同文件夹的配对，并且获得excel文件的相对路径
        input_name = self.input_path[item]
        input_item_path = os.path.join(self.input_dir, input_name)
        output_item_path = os.path.join(self.output_dir, input_name)

        # 读取excel文件的矩阵，并且转为张量形式
        input_tensor = excel_to_matrix(input_item_path).excel_to_matrixx()
        output_tensor = excel_to_matrix(output_item_path).excel_to_matrixx()

        return input_tensor, output_tensor


# mydata = SEGData('inputfigure96pixel', 'outputmatrix96pixel')
# input_tensor, output_tensor = mydata[2]
#
# toPIL = transforms.ToPILImage()
# inpic = toPIL(input_tensor)
# inpic.show()
#
# outpic = toPIL(output_tensor)
# outpic.show()



