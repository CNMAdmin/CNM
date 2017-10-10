import os
import autoencoder as ae
import trainer
import numpy as np

path = 'G:/Richard/new/'
def representation():
  s, f = ae.load_data('10 ICP int/')
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
  total_rep_imgs = ae.autoencoding_cnn(total_image, total_image, fold='tot')
  for l, v in file_len_map.items():
    npy_ori = []
    npy_rep = []
    for m in range(v[0], v[1]):
      npy_ori.append(np.array(total_image[m]).reshape(64, 64, 1))
      npy_rep.append(np.array(total_rep_imgs[m]).reshape(64, 64, 1))
    np.save('npy/icp/tot/'+l+'.ori.npy', np.array(npy_ori))
    np.save('npy/icp/tot/'+l+'.rep.npy', np.array(npy_rep))


def np_transform_error_deal(t):
  n = np.array([])
  for v in t:
    n = np.append(n, np.array(v))
  return n

def cnn_train():
  train, test = trainer.read_total_module()
  #train[0] = np_transform_error_deal(train[0])
  model = trainer.create_model(64)
  print(model.summary())
  model.fit(np.array(train[0]), np.array(train[1]), epochs=10)
  model.save('net/CNN/total_CNN_ICP10.net')
  
if __name__=='__main__':
  os.chdir(path)
  representation()
  cnn_train()
  print('Finish!')