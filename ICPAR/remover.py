from keras.models import load_model
from trainer import read_1_file
import numpy as np

samples = ['15', '31', '40', '45', '53', '70', '131', '195', '248', '299']
if __name__=='__main__':
  model = load_model('net/CNN/total_CNN10.net') # network
  for v in samples:
    x, y, tl = read_1_file(v+'.abp.fin.npy', "total") # load file
    pen = open('result/'+v+'.prd.fin.csv', 'w') # ar csv
    prd = model.predict(np.array(x))
    for i in range(len(prd)):
      pen.write(str(i)+ ',' + str(round(prd[i][0])) + '\n')
    pen.close()
  print('fin')
