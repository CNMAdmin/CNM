import os
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/Richard/ICP artifact rejection using representation/plateau 101 first_ICP_filter/'
if __name__=='__main__':
  #files = os.listdir(path)
  files = ['D:/Source code/Git/CNM/CNM/ICPAR/int/70_1_int_rev.csv']
  for f in files:
    #if f.split('.')[0].split('_')[-1] == 'raw':
      fi = open(f)
      lines = fi.readlines()
      fi.close()
      if not os.path.exists('img/icp/'+f.split('_')[0].split('/')[-1]):
        os.mkdir('img/icp/'+f.split('_')[0].split('/')[-1])
      i = 0
      for line in lines:
        sl = line.split(',')
        id = sl[0]
        cur_sig = []
        for s in sl[1:]:
          cur_sig.append(float(s))
        x = np.linspace (0, 0.8, len(cur_sig))
        plt.scatter(x, cur_sig)
        plt.savefig('img/icp/'+f.split('_')[0].split('/')[-1] + '/' + id + '.icp.png')
        plt.cla(); plt.clf()
        if i % 10 == 0:
          print(str(i) + ' / ' + str(len(lines)))
        i+=1