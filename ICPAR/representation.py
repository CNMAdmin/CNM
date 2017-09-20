import os 
import numpy as np
import matplotlib.pyplot as plt
import autoencoder as ae
import sys
sys.setrecursionlimit(100000)
path = 'D:/Richard/ICP artifact rejection using representation/Artifact labeled ABP/int3/'

if __name__ == '__main__':
  signals, filenames = ae.load_data(path)
  total_image = []
  total_signal =[]
  file_len_map = {}

  bef_cnt = 0
  for i in range(len(filenames)):
    filename = filenames[i].split('_')[0]
    imgs = ae.signal_to_img(signals[i])
    total_image.extend(imgs)
    total_signal.extend(signals[i])
    file_len_map[filename] = [bef_cnt, bef_cnt + len(imgs)]
    bef_cnt += len(imgs)
  total_rep_imgs = ae.autoencding_cnn_using_net(total_image)
  for k, v in file_len_map.items():
    npy_arr = []
    for i  in range(v[0], v[1]):
      plt.figure(1)
      plt.imshow(total_image[i].reshape(64, 64))
      plt.savefig('img/ori_a/'+k+'_'+str(i-v[0])+'.png')
      plt.cla(); plt.clf()

      plt.figure(1)
      plt.imshow(total_rep_imgs[i].reshape(64, 64))
      plt.savefig('img/rep_a/'+k+'_'+str(i-v[0])+'.png')
      plt.cla(); plt.clf()

      npy_arr.append(total_rep_imgs[i])
    print(k + ' fin')
    np.save('npy/'+k+'.abp.t.npy', npy_arr)
