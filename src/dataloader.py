import numpy as np
import torch
import nibabel as nib
import random
from nilearn.image import resample_img


class MRIdata3D(Dataset):
  def __init__(self, img_size, data_path, label_path, file_names):
    label = label_path   ##ydata
    self.img_size = img_size
    self.label=label.set_index('Subject_id')
    self.file_paths = file_names  ##file_list
    self.transforms = transforms.Compose([transforms.ToTensor()])
    self.train_path = data_path

  def __getitem__(self, idx):
    img, name = self.read_img(idx)
    img_tensor=[]

    for d in range(img.shape[-1]):
      img2d = Image.fromarray(img[...,d])
      img_tensor2d = self.transforms(img2d) #1,h,w
      img_tensor.append(img_tensor2d)
    img_tensor = torch.cat(img_tensor,axis=0) #d,h,w
    d,h,w=img_tensor.size()
    img_tensor = torch.unsqueeze(img_tensor, 0) #1,d,h,w

    age = self.label.loc[name, 'age']
 #   print(img_tensor.shape)
    age = torch.Tensor(np.array(age))
    return img_tensor,name, age

  def read_img(self, idx):
    proxy = nib.load(self.train_path+self.file_paths[idx])
    image_t = resample_img(resample_img(
            img=proxy,
            target_affine=np.diag([1.2,1.2,1.2]),
            interpolation='continuous'
            ))
    # arr = proxy.dataobj[..., ridx:ridx+self.img_size]
    arr = image_t.get_fdata()
    arr = np.array(arr)
    
    arr = np.nan_to_num(arr) #nan valued pixel 2 zero
    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr)) #normalize
    name = os.path.basename(self.file_paths[idx]).split('_')[0] #ex) 'BP0001'
    
    return arr, name

  def __len__(self):
    return len(self.file_paths)#,int(100*bs/0.8)) # # of iteration (__len__/batch_size*train_ratio)
