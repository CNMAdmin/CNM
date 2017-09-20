import os 
import numpy as np
import matplotlib.pyplot as plt
import autoencoder as ae
import sys
sys.setrecursionlimit(100000)

fold1 = 'fold/icp/0/'
fold2 = 'fold/icp/1/'
fold3 = 'fold/icp/2/'
fold4 = 'fold/icp/3/'
fold5 = 'fold/icp/4/'

if __name__ == '__main__':
  s_1, f_1 = ae.load_data(fold1)
  s_2, f_2 = ae.load_data(fold2)
  s_3, f_3 = ae.load_data(fold3)
  s_4, f_4 = ae.load_data(fold4)
  s_5, f_5 = ae.load_data(fold5)
  signals = [s_1, s_2, s_3, s_4, s_5]
  filenams = [f_1, f_2, f_3, f_4, f_5]
  for i in range(0, 2):
    print(str(i) + ' fold start!')
    s = []; f = []; sa = []; fa =[];
    for j  in range(5):
      if i == j:
        sa.extend(signals[j])
        fa.extend(filenams[j])
      else:
        s.extend(signals[j])
        f.extend(filenams[j])

    total_image = []
    total_signal = []
    file_len_map = {}

    bef_cnt = 0
    for k in range(len(f)):
      filename = f[k].split('_')[0]
      imgs = ae.signal_to_img(s[k])
      total_image.extend(imgs)
      total_signal.extend(s[k])
      file_len_map[filename] = [bef_cnt, bef_cnt + len(imgs)]
      bef_cnt += len(imgs)

    total_rep_imgs = ae.autoencoding_cnn(total_image, total_image, fold=i)
    
    for l, v in file_len_map.items():
      npy_ori = []
      npy_rep = []
      for m in range(v[0], v[1]):
        npy_ori.append(total_image[m])
        npy_rep.append(total_rep_imgs[m])
        
      np.save('npy/icp/'+str(i)+'/'+l+'.ori.npy', npy_ori)
      np.save('npy/icp/'+str(i)+'/'+l+'.rep.npy', npy_rep)
      
    total_image = []
    total_signal = []
    file_len_map = {}

    bef_cnt = 0
    for k in range(len(fa)):
      filename = fa[k].split('_')[0]
      imgs = ae.signal_to_img(sa[k])
      total_image.extend(imgs)
      total_signal.extend(sa[k])
      file_len_map[filename] = [bef_cnt, bef_cnt + len(imgs)]
      bef_cnt += len(imgs)
    total_rep_imgs = ae.autoencding_cnn_using_net(total_image, fold=i)
    
    for l, v in file_len_map.items():
      npy_ori = []
      npy_rep = []
      for m in range(v[0], v[1]):
        npy_ori.append(total_image[m])
        npy_rep.append(total_rep_imgs[m])

      np.save('npy/icp/'+str(i)+'/'+l+'.ori.non.npy', npy_ori)
      np.save('npy/icp/'+str(i)+'/'+l+'.rep.non.npy', npy_rep)
    print(str(i) + ' fold finished!')

