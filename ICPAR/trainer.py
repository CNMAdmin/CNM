import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import keras.backend as K
import random
import os
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(1000000)


def data_load_module(tf):
  file = open('int/' + tf + '_1_int_rev.csv')
  lines = file.readlines()
  file.close()
  arr = np.load('npy/' + tf + '.icp.t.npy')
  x = []; y = []; tl = []
  for line in lines:
    sl = line.split(',')
    sid = int(sl[0])
    #if float(sl[2]) > 60: continue
    if int(sl[1]) == 1:
      y.append([1, 0])
    else:
      y.append([0, 1])
    tl.append(float(sl[2]))
    x.append(arr[sid])
  return x, y, tl

def rejection(x, y, tl):
  pos_idx = []
  neg_idx = []
  for i in range(len(y)):
    if y[i][0] == 0:
      pos_idx.append(i)
    else:
      neg_idx.append(i)
  
  lp = len(pos_idx)
  ln = len(neg_idx)

  acc_cnt = lp / ln if lp > ln else ln / lp

  tot_idx = []
  if lp > ln:
    tot_idx = pos_idx
    for i in range(int(acc_cnt)):
      tot_idx.extend(neg_idx)
  else:
    tot_idx = neg_idx
    for i in range(int(acc_cnt)):
      tot_idx.extend(pos_idx)
  random.shuffle(tot_idx)
  new_x = []
  new_y = []
  new_tl = []
  for idx in tot_idx:
    new_x.append(x[idx])
    new_y.append(y[idx])
    new_tl.append(tl[idx])
  return new_x, new_y, new_tl

def data_load(train_list, test_list):
  train_x = []; train_y = []; train_tl = []
  for tf in train_list:
    x, y, tl = data_load_module(tf)
    train_x.extend(x); train_y.extend(y); train_tl.extend(tl)
  train_x, train_y, train_tl = rejection(train_x, train_y, train_tl)
  test_x = []; test_y = []; test_tl = []
  for tf in test_list:
    x, y, tl = data_load_module(tf)
    test_x.extend(x); test_y.extend(y); test_tl.extend(tl)

  return train_x, train_y, train_tl, test_x, test_y, test_tl

def fold_data_load(i):
  train_x = []; train_y = []; train_tl = []

def create_model(ipt_dim):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(ipt_dim, ipt_dim, 1)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
  return model

def performance_generator(tp, tn, fp, fn):
  sen = tp / (tp + fn) if (tp + fn) > 0 else 0
  spe = tn / (tn + fp) if (tn + fp) > 0 else 0
  ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
  npv = tn / (tn + fn) if (tn + fn) > 0 else 0
  npd = (sen + spe) / 2
  acc = (tp + tn) / (tp + tn + fp + fn)

  return [sen, spe, ppv, npv, npd, acc]

def counter(y):
  pc = 0; nc = 0
  for i in range(len(y)):
    if round(y[i][0]) == 0:
      pc += 1
    else:
      nc += 1
  return pc, nc


def get_pred_perfomance(test_y, pred_y, time_line, test_x):
  tp = 0; tn = 0; fp = 0; fn = 0;
  tpt = 0; tnt = 0; fpt = 0; fnt = 0;
  for i in range(len(pred_y)):
    cp = round(pred_y[i][0])
    ca = test_y[i][0]
    if cp == ca:
      if cp == 0:
        tp += 1
        tpt += time_line[i]
        
      else:
        tn += 1
        tnt += time_line[i]
        
    else:
      if cp == 0:
        fp += 1
        fpt += time_line[i]
        
      else:
        fn += 1
        fnt += time_line[i]

  ca = performance_generator(tp, tn, fp, fn)
  ta = performance_generator(tpt, tnt, fpt, fnt)

  cs = str(tp) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn)
  for v in ca:
    cs += ',' + str(v)
  ts =  str(tpt) + ',' + str(tnt) + ',' + str(fpt) + ',' + str(fnt)
  for v in ta:
    ts += ',' + str(v)

  print('Count:' + cs)
  print('Time:' + ts)

  return cs  + ',' + ts

def read_1_file(file, pos):
  pid = file.split('.')[0].split('/')[-1]
  f = open('10 ABP int/'+pid+'_1_int.csv')
  lines = f.readlines()
  f.close()
  arr = np.load('npy/abp/'+str(pos)+'/'+file)
  x = []; y = []; tl = [];
  for line in lines:
    sl = line.split(',')
    sid = int(sl[0])
    cur_y = [0, 0]
    if int(sl[1]) == 1:
      cur_y = [1, 0]
    else:
      cur_y = [0, 1]
    if float(sl[2]) > 1.5 or float(sl[2]) < 0.2:
      cur_y = [1, 0]
    y.append(cur_y)
    tl.append(float(sl[2]))
    x.append(arr[sid])
  return x, y, tl

def read_1_icp_file(file, pos):
  pid = file.split('.')[0].split('/')[-1]
  f = open('10 ICP int/'+pid+'_1_int.csv')
  lines = f.readlines()
  f.close()
  arr = np.load('npy/icp/'+str(pos)+'/'+file)
  x = []; y = []; tl = [];
  for line in lines:
    sl = line.split(',')
    sid = int(sl[0])
    if int(sl[1]) == 1:
      y.append([1, 0])
    else:
      y.append([0, 1])
    tl.append(float(sl[2]))
    x.append(arr[sid])
  return x, y, tl

def time_based_amplitude(train):
  new_train = [[], [], []]
  for i in range(len(train[2])):
    for j in range(round(train[2][i]+0.5)):
      new_train[0].append(train[0][i])
      new_train[1].append(train[1][i])
      new_train[2].append(train[2][i])
  return new_train

def agumentation(train):
  new_train = [[], [], []]
  pos_s = [[], []]
  neg_s = [[], []]
  pos_cnt = 0; neg_cnt = 0;
  for i in range(len(train[1])):
    if train[1][i][0] == 0:
      pos_cnt += 1
      pos_s[0].append(train[0][i])
      pos_s[1].append(train[2][i])
    else:
      neg_cnt += 1
      neg_s[0].append(train[0][i])
      neg_s[1].append(train[2][i])
  for k in range(0, int(pos_cnt/neg_cnt)):
    for i in range(0, len(neg_s[0])):
      new_train[0].append(neg_s[0][i])
      new_train[1].append([1, 0])
      new_train[2].append(neg_s[1][i])
  for i in range(0, len(pos_s[0])):
    new_train[0].append(pos_s[0][i])
    new_train[1].append([0, 1])
    new_train[2].append(pos_s[1][i])
  return new_train

def read_module(pos):
  files = os.listdir('npy/abp/' + str(pos))
  test_x = []; test_y = []; test_tl = [];
  train_x = []; train_y = []; train_tl = [];
  for file in files:
    if 'rep' in file:
      if 'non' in file:
        x, y, tl = read_1_file(file, pos)
        test_x.extend(x); test_y.extend(y); test_tl.extend(tl)
      else:
        x, y, tl = read_1_file(file, pos)
        train_x.extend(x); train_y.extend(y); train_tl.extend(tl)
  return [train_x, train_y, train_tl], [test_x, test_y, test_tl]

def read_icp_module(pos):
  files = os.listdir('npy/icp/' + str(pos))
  test_x = []; test_y = []; test_tl = [];
  train_x = []; train_y = []; train_tl = [];
  for file in files:
    if 'rep' in file:
      if 'non' in file:
        x, y, tl = read_1_icp_file(file, pos)
        test_x.extend(x); test_y.extend(y); test_tl.extend(tl)
      else:
        x, y, tl = read_1_icp_file(file, pos)
        train_x.extend(x); train_y.extend(y); train_tl.extend(tl)
  return [train_x, train_y, train_tl], [test_x, test_y, test_tl]

def read_total_module():
  files = os.listdir('npy/abp/tot/')
  train_x = []; train_y = []; train_tl = [];
  for file in files:
    x, y, tl = read_1_file(file, "tot")
    train_x.extend(x); train_y.extend(y); train_tl.extend(tl)
  return [train_x, train_y, train_tl], [train_x, train_y, train_tl]

def pred_revise(pred, tl):
  for i in range(len(pred)):
    if tl[i] > 1.5 or tl[i] < 0.2:
      pred[i] = [1, 0]
  return pred

if __name__ =='__main__':
  path = 'G:/Richard/1018/'
  os.chdir(path)
  setting = 'icp_aug'
  pos = 1
  print(str(pos))
  train, test = read_icp_module(pos)
  #train, test = read_total_module()
  #train = time_based_amplitude(train)
  train = agumentation(train)
  model = create_model(64)
  print(model.summary())
  model.fit(np.array(train[0]), np.array(train[1]), validation_data=(np.array(test[0]), np.array(test[1])), epochs=10)
  model.save('net/CNN/'+str(pos)+'_' +setting+'_CNN10.net')
  pred = model.predict(np.array(test[0]))
  pred = pred_revise(pred, test[2])
  pen = open('test/'+str(pos) + '_' + setting + '.csv', 'w')
  pen.write('idx,real,pred\n')
  for i in range(len(test[1])):
    pen.write(str(i) + ',' + str(test[1][i][0]) + ',' + str(pred[i][0]) + '\n')
  pen.close()

  sentence = get_pred_perfomance(test[1], pred, test[2], test[0])
  pen = open('CNN_result.csv', 'a')
  pen.write('\n' + str(pos) + '_' + setting + ',' + sentence)
  pen.close()
