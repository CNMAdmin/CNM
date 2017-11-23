from keras.models import load_model
from trainer import read_1_file
import numpy as np
import os


path = 'G:/Richard/1018/'
samples = os.listdir('G:/Richard/ABP_ICP_AR_SIGNALS/plateau/ABP_int/')

if __name__=='__main__':
  os.chdir(path)
  model = load_model('net/CNN/total_CNN_ABP10.net') # network
  for v in samples:
    x, y, tl = read_1_file(v+'.rep.npy', "tot") # load file
    pen = open('result/abp/'+v+'.prd.fin.csv', 'w') # ar csv
    prd = model.predict(np.array(x))
    for i in range(len(prd)):
      pen.write(str(i)+ ',' + str(round(prd[i][0])) + '\n')
    pen.close()
  print('fin')
