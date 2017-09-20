import os
import numpy as np
import matplotlib.pyplot as plt
import autoencoder as ae
import sys
sys.setrecursionlimit(100000)
path = 'int/icp/'

if __name__=='__main__':
  """
  files = os.listdir(path)
  for f in files:
    sig, lab = ae.load_one_data(path + f)
    if not os.path.exists('img/ori/'+f.split('_'[0])):
      os.mkdir('img/ori/'+f.split('_')[0])
    if not os.path.exists('img/rep/'+f.split('_'[0])):
      os.mkdir('img/rep/'+f.split('_')[0])
    for i in range(len(lab)):
      cur_img = ae.signal_to_img(sig[i])
  """
  signals, filenames = ae.load_data(path)
  total_image = []
  total_signal = []
  file_len_map = {}
 
  bef_cnt = 0
  for i in range(len(filenames)):
    filename = filenames[i].split('_')[0]
    imgs = ae.signal_to_img(signals[i])
    total_image.extend(imgs)
    total_signal.extend(signals[i])
    file_len_map[filename] = [bef_cnt, bef_cnt + len(imgs)]
    bef_cnt += len(imgs)
  total_rep_imgs = ae.autoencoding_cnn(total_image, total_image, fold = 'final', img_dim=64, encoding_dim=16)
  
  for k, v in file_len_map.items():
    npy_arr = []
    for i in range(v[0], v[1]):
      """
      plt.figure(1)
      plt.imshow(total_image[i].reshape(64, 64))
      plt.savefig('img/ori_b/'+k+'_'+str(i-v[0])+'.png')
      plt.cla(); plt.clf()
      
      plt.figure(1)
      plt.imshow(total_rep_imgs[i].reshape(64, 64))
      plt.savefig('img/rep_a/'+k+'_'+str(i-v[0])+'.png')
      """
      npy_arr.append(total_rep_imgs[i])
    np.save('npy/'+k+'.abp.fin.npy', npy_arr)
