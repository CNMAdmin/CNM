from keras.models import load_model
from trainer import read_1_file
import numpy as np
import os

samples = ['15', '31', '40', '45', '53', '70', '131', '195', '248', '299']
path = 'G:/Richard/new/'
if __name__=='__main__':
  os.chdir(path)
  model = load_model('net/CNN/total_CNN_ICP10.net') # network
  for v in samples:
    x, y, tl = read_1_file(v+'.rep.npy', "tot") # load file
    pen = open('result/icp/'+v+'.prd.fin.csv', 'w') # ar csv
    prd = model.predict(np.array(x))
    for i in range(len(prd)):
      pen.write(str(i)+ ',' + str(round(prd[i][0])) + '\n')
    pen.close()
  print('fin')
