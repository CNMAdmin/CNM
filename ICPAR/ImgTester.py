import numpy as np
import os
import matplotlib.pyplot as plt

path = 'G:/Richard/new/'
os.chdir(path)
poses = ['0', '1', '2', '3']
for pos in poses:
  print(str(pos))
  files = os.listdir('npy/abp/' + str(pos))
  for file in files:
    print(file)
    pid = file.split('.')[0].split('/')[-1]
    f = open('10 ABP int/'+pid+'_1_int.csv')
    lines = f.readlines()
    f.close()

    arr = np.load('npy/abp/'+str(pos)+'/'+file)
    x = []; y = []; tl = [];
    id = 0
    for line in lines:
      sl = line.split(',')
      sid = int(sl[0])
      
      plt.figure(id)
      plt.imshow(arr[sid].reshape(64, 64))
      plt.savefig('img/abp/'+str(pos)+'/'+file+'_'+str(id)+'_'+sl[1]+'.png')
      plt.cla(); plt.clf()      
      id += 1

