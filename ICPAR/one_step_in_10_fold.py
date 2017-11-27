import os
import autoencoder as ae
import numpy as np
import trainer
from keras.models import load_model

def gen_full_train_network(isABP):
  signal_tag = 'ABP' if isABP else 'ICP'
  int_dat = os.listdir('10 ' + signal_tag + ' int')
  train_x = []; train_y = []; train_t = [];
  for int_d in int_dat:
    sig, lab, tim = ae.load_one_data('10 ' + signal_tag + ' int/' + int_d)
    train_x.extend(sig)
    train_y.extend(lab)
    train_t.extend(tim)
  train_x = ae.signal_to_img(train_x)
  train_rep = ae.autoencoding_cnn(train_x, train_x, 'net/FINAL_'+signal_tag+'.net')
  new_train = augmentation(train_rep, label_gen(train_y), train_t)
  model = trainer.create_model(64)
  print(model.summary())
  model.fit(np.array(new_train[0]), np.array(new_train[1]), epochs=10, shuffle=True)
  model.save('net/CNN/FINAL_'+signal_tag+'.net')


def remove_artifact(isABP, int_path):
  samples = os.listdir(int_path)
  signal_tag = 'ABP' if isABP else 'ICP'
  rep_model = load_model('net/FINAL_' + signal_tag + '.net')
  cla_model = load_model('net/CNN/FINAL_' + signal_tag + '.net')
  for v in samples:
    sig, lab, tim = ae.load_one_data(int_path + '/' + v)
    train_x = ae.signal_to_img(train_x)
    train_rep_x = ae.autoencding_cnn_using_net(train_x, i)

def gen_rep_img(isABP):
  if isABP:
    int_dat = os.listdir('10 ABP int')
  else:
    int_dat = os.listdir('10 ICP int')
  file_names = []
  total_sig = []
  total_lab = []
  total_tim = []
  for int_d in int_dat:
    if isABP:
      sig, lab, tim = ae.load_one_data('10 ABP int/' + int_d)
    else:
      sig, lab, tim = ae.load_one_data('10 ICP int/' + int_d)
    total_sig.append(sig)
    total_lab.append(lab)
    total_tim.append(tim)
    file_names.append(int_d)
  for i in range(len(total_sig)):
    signal_tag = 'ABP' if isABP else 'ICP'
    if os.path.isfile('npy10/' + signal_tag + '/' + file_names[i] + '.train.ori.npy'): continue
    print(i)
    train_x = []; train_y = []; train_t = [];
    test_x = []; test_y = []; test_t = [];
    for j in range(len(total_sig)):
      if i == j:
        test_x.extend(total_sig[j])
        test_y.extend(total_lab[j])
        test_t.extend(total_tim[j])
      else:
        train_x.extend(total_sig[j])
        train_y.extend(total_lab[j])
        train_t.extend(total_tim[j])
    train_x = ae.signal_to_img(train_x)
    test_x = ae.signal_to_img(test_x)
    train_rep_x = ae.autoencoding_cnn(train_x, train_x, 'net/'+str(i)+'_'+signal_tag+'.net')
    test_rep_x = ae.autoencding_cnn_using_net(test_x, 'net/'+str(i)+'_'+signal_tag+'.net')
    np.save('npy10/' + signal_tag + '/' + file_names[i] + '.train.ori.npy', train_x)
    np.save('npy10/' + signal_tag + '/' + file_names[i] + '.train.rep.npy', train_rep_x)
    np.save('npy10/' + signal_tag + '/' + file_names[i] + '.train.etc.npy', [train_y, train_t])
    np.save('npy10/' + signal_tag + '/' + file_names[i] + '.test.ori.npy', test_x)
    np.save('npy10/' + signal_tag + '/' + file_names[i] + '.test.rep.npy', test_rep_x)
    np.save('npy10/' + signal_tag + '/' + file_names[i] + '.test.etc.npy', [test_y, test_t])

def gen_svm_img(isABP):
  signal_tag = 'ABP' if isABP else 'ICP'
  int_dat = os.listdir('10 ' + signal_tag + ' int')
  file_names = [];  total_sig = [];  total_lab = [];  total_tim = []
  for int_d in int_dat:
    sig, lab, tim = ae.load_one_data('10 ' + signal_tag + ' int/' + int_d)
    total_sig.append(sig); total_lab.append(lab); total_tim.append(tim); file_names.append(int_d)
  for i in range(len(total_sig)):
    if os.path.isfile('npy10/' + signal_tag + '_S/' + file_names[i] + '.train.ori.npy'): continue
    print(i)
    train_x = []; train_y = []; train_t = [];
    test_x = []; test_y = []; test_t = [];
    for j in range(len(total_sig)):
      if i == j:
        test_x.extend(total_sig[j])
        test_y.extend(total_lab[j])
        test_t.extend(total_tim[j])
      else:
        train_x.extend(total_sig[j])
        train_y.extend(total_lab[j])
        train_t.extend(total_tim[j])
    np.save('npy10/' + signal_tag + '_S/' + file_names[i] + '.train.ori.npy', train_x)
    np.save('npy10/' + signal_tag + '_S/' + file_names[i] + '.train.etc.npy', [train_y, train_t])
    np.save('npy10/' + signal_tag + '_S/' + file_names[i] + '.test.ori.npy', test_x)
    np.save('npy10/' + signal_tag + '_S/' + file_names[i] + '.test.etc.npy', [test_y, test_t])

def label_gen(lab):
  new_lab = []
  for l in lab:
    if l == 0:
      new_lab.append([0, 1])
    else:
      new_lab.append([1, 0])
  return new_lab

def label_back(lab):
  new_lab = []
  for l in lab:
    if l[0] == 0:
      new_lab.append(0)
    else:
      new_lab.append(1)
  return new_lab

def augmentation(rep, lab, tl):
  new_train = [[], [], []]
  pos_s = [[], []]
  neg_s = [[], []]
  pos_cnt = 0; neg_cnt = 0;
  for i in range(len(lab)):
    if lab[i][0] == 0:
      pos_cnt += 1
      pos_s[0].append(rep[i])
      pos_s[1].append(tl[i])
    else:
      neg_cnt += 1
      neg_s[0].append(rep[i])
      neg_s[1].append(tl[i])
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

def n_augmentation(rep, lab, tl):
  return [rep, lab, tl]



def test(isABP):
  signal_tag = 'ABP' if isABP else 'ICP'
  int_dat = os.listdir('10 ' + signal_tag + ' int')
  file_names = []
  setting = "10 fold nonaug " + signal_tag
  print(setting)
  for int_d in int_dat:
    file_names.append(int_d)
  for i in range(0, 10):
    train_ori = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.train.ori.npy')
    train_rep = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.train.rep.npy')
    train_etc = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.train.etc.npy')
    test_ori = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.test.ori.npy')
    test_rep = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.test.rep.npy')
    test_etc = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.test.etc.npy')
    model = trainer.create_model(64)
    print(model.summary())
    new_train = n_augmentation(train_rep, label_gen(train_etc[0]), train_etc[1])
    model.fit(np.array(new_train[0]), np.array(new_train[1]), validation_data=(np.array(test_rep), np.array(label_gen(test_etc[0]))), epochs=10, shuffle=True)
    model.save('net/CNN/'+str(i)+'_' +setting+'_CNN10_' + signal_tag + '.net')
    pred = model.predict(np.array(test_rep))
    pen = open('test/' + signal_tag + '/' + str(file_names[i]) + '_' + setting + '.csv', 'w')
    pen.write('idx,real,pred\n')
    for j in range(len(label_gen(test_etc[0]))):
      pen.write(str(j) + ',' + str(test_etc[0][j]) + ',' + str(pred[j][0]) + '\n')
    pen.close()
    sentence = trainer.get_pred_perfomance(label_gen(test_etc[0]), pred, test_etc[1], test_rep)
    pen = open('CNN_result.csv', 'a')
    pen.write('\n' + str(file_names[i]) + '_' + setting + ',' + sentence)
    pen.close()
def ori_tfr(t):
  tt = t.reshape(len(t), 64, 64, 1).astype('float32') / 255
  return tt

def test_ori(isABP):
  signal_tag = 'ABP' if isABP else 'ICP'
  int_dat = os.listdir('10 ' + signal_tag + ' int')
  file_names = []
  setting = "10 fold aug ori " + signal_tag
  print(setting)
  for int_d in int_dat:
    file_names.append(int_d)
  for i in range(0, 10):
    train_ori = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.train.ori.npy')
    #train_rep = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.train.rep.npy')
    train_etc = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.train.etc.npy')
    test_ori = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.test.ori.npy')
    test_etc = np.load('npy10/' + signal_tag + '/' + file_names[i] + '.test.etc.npy')
    model = trainer.create_model(64)
    print(model.summary())
    new_train = augmentation(ori_tfr(train_ori), label_gen(train_etc[0]), train_etc[1])
    model.fit(np.array(new_train[0]), np.array(new_train[1]), validation_data=(np.array(ori_tfr(test_ori)), np.array(label_gen(test_etc[0]))), epochs=10, shuffle=True)
    model.save('net/CNN/'+str(i)+'_' +setting+'_CNN10_' + signal_tag + '.net')
    pred = model.predict(np.array(ori_tfr(test_ori)))
    pen = open('test/' + signal_tag + '/' + str(file_names[i]) + '_' + setting + '.csv', 'w')
    pen.write('idx,real,pred\n')
    for j in range(len(label_gen(test_etc[0]))):
      pen.write(str(j) + ',' + str(test_etc[0][j]) + ',' + str(pred[j][0]) + '\n')
    pen.close()
    sentence = trainer.get_pred_perfomance(label_gen(test_etc[0]), pred, test_etc[1], test_rep)
    pen = open('CNN_result.csv', 'a')
    pen.write('\n' + str(file_names[i]) + '_' + setting + ',' + sentence)
    pen.close()

def pred_test(Y_pred, Y_test, tl):
  tp = 0; tn = 0; fp = 0; fn = 0
  tpc = 0; tnc = 0; fpc = 0; fnc = 0
  for i in range(0, len(Y_pred)):
    if Y_test[i] == 0:
      if round(Y_pred[i]) == 0:
        tpc += 1
        tp += tl[i]
      else:
        fnc += 1
        fn += tl[i]
    else:
      if round(Y_pred[i]) == 0:
        fpc += 1
        fp += tl[i]
      else:
        tnc += 1
        tn += tl[i]
  ca = trainer.performance_generator(tpc, tnc, fpc, fnc)
  ta = trainer.performance_generator(tp, tn, fp, fn)

  cs = str(tpc) + ',' + str(tnc) + ',' + str(fpc) + ',' + str(fnc)
  for v in ca:
    cs += ',' + str(v)
  ts =  str(tp) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn)
  for v in ta:
    ts += ',' + str(v)

  print('Count:' + cs)
  print('Time:' + ts)

  return cs  + ',' + ts

def test_lsvm(isABP, C=1.0):
  import time
  start_total = time.time()
  signal_tag = 'ABP' if isABP else 'ICP'
  int_dat = os.listdir('10 ' + signal_tag + ' int')
  file_names = []
  setting = "10 fold lsvm " + str(C) + ' ' + signal_tag
  print(setting)
  for int_d in int_dat:
    file_names.append(int_d)
  for i in range(0, 10):
    start_fold = time.time()
    print(str(i) + ' fold start!')
    train = np.load('npy10/' + signal_tag + '_S/' + file_names[i] + '.train.ori.npy')
    train_etc = np.load('npy10/' + signal_tag + '_S/' + file_names[i] + '.train.etc.npy')
    test = np.load('npy10/' + signal_tag + '_S/' + file_names[i] + '.test.ori.npy')
    test_etc = np.load('npy10/' + signal_tag + '_S/' + file_names[i] + '.test.etc.npy')

    from sklearn import svm

    model = svm.SVC(C=C)
    print('Training....')
    model.fit(np.array(train), np.array(train_etc[0]))
    #model.save('net/CNN/'+str(i)+'_' +setting+'_CNN10_' + signal_tag + '.net')
    pred = model.predict(np.array(test))
    pen = open('test/' + signal_tag + '/' + str(file_names[i]) + '_' + setting + '.csv', 'w')
    pen.write('idx,real,pred\n')
    for j in range(len(label_gen(test_etc[0]))):
      pen.write(str(j) + ',' + str(test_etc[0][j]) + ',' + str(pred[j]) + '\n')
    pen.close()
    sentence = pred_test(pred, test_etc[0], test_etc[1])
    pen = open('CNN_result.csv', 'a')
    pen.write('\n' + str(file_names[i]) + '_' + setting + ',' + sentence)
    pen.close()
    print(time.time() - start_fold + ' sec for this fold')
  print(time.time() - start_total + ' sec for all process')

def make_img():
  import matplotlib.pyplot as plt
  int_dat = os.listdir('10 ICP int')
  for int_d in int_dat:
    sig, lab, tim = ae.load_one_data('10 ICP int/' + int_d)
    img = ae.signal_to_img(sig)
    if not os.path.isdir('img/' + int_d.split('_')[0]):
      os.mkdir('img/' + int_d.split('_')[0])
      os.mkdir('img/' + int_d.split('_')[0] + '/nor')
      os.mkdir('img/' + int_d.split('_')[0] + '/art')
    else:
      continue
    for i in range(len(tim)):
      plt.figure(1)
      plt.imshow(img[i].reshape(64, 64))
      if lab[i] == 0:
        plt.savefig('img/' + int_d.split('_')[0] + '/nor/' + str(i) + '.png')
      else:
        plt.savefig('img/' + int_d.split('_')[0] + '/art/' + str(i) + '.png')
      plt.cla(); plt.clf()

if __name__ =='__main__':
  path = 'F:/Richard/1018'
  os.chdir(path)
  #make_img()
  #gen_rep_img(isABP=True)
  #test(isABP=True)
  #test_ori(isABP=True)
  test_lsvm(isABP=True, C=0.01)
  #gen_full_train_network()
  #gen_svm_img(isABP=True)
  #gen_svm_img(isABP=False)

    
    