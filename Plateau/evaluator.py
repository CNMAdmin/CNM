import fileController as fc
import modelController as mc
import numpy as np
import labelling as lb
import preprocess as pp
import csv
import datetime
import matplotlib.pyplot as plt
import pandas as pd

def eval_model(model,env):
    testlist=env.file["test_file_list"]
    label_path=env.get_config("path","label_path")
    labeldata=lb.load_label(label_path)
    feature=env.get_config("data","feature",type="list")

    save_result=env.get_config("test","save_result",type="int")
    save_graph=env.get_config("test","save_graph",type="int")
    save_time=env.get_config("test","save_time",type="int")
    segment_test=env.get_config("test","segment_test",type="int")
    hour_limit=env.get_config("test","hour_limit",type="int")

    confusion=np.zeros(4)

    for fileidx in range(len(testlist)):
        filepath=testlist[fileidx]
        data=fc.get_data(filepath,feature)
        print(data)
        dataX=pp.transform_data(data)
        print(dataX)
        true=lb.data_labeling(dataX,filepath,labeldata)
        pred=mc.predict_data(dataX,model,env,pp=0)

        if save_graph==1:
            idx_group=pp.cut_idx_by_hour(dataX,hour_limit=hour_limit)
            figure_num=draw_graph(dataX,true,pred,fileidx,env,idx_group)        
        if save_result==1:
            if segment_test==1:
                confusion=confusion+get_segment_confusion(true,pred)
            else:
                confusion=confusion+get_confusion(true,pred)
        if save_time==1:
            print("Get Prediction time")
            pred_times=save_pred_time(pred,dataX[0],env)
            print("Write to csv")
            write_pred_data(data,pred_times,filepath,env)
    if save_result==1:
        if segment_test==1:
            segment_confusion_result(confusion,env)
        else:
            confusion_result(confusion,env)

def write_pred_data(data,times,filename,env):
    result_path=env.get_config("path","result_path")

    timelapse=data[0]
    icps=data[1]
    abps=data[2]

    filename=filename[filename.find("\\")+1:filename.find(".csv")]
    print(filename)
    result=np.empty((0,3))
    print(len(timelapse))

    time_idx=0
    progress=0
    max_time=len(times)
    print(max_time)
    for i in range(len(timelapse)):
        if i%100000==0:
            print(i)
        start_time=times[time_idx][0]
        end_time=times[time_idx][1]
        cur_time=timelapse[i]
        cur_icp=icps[i]
        cur_abp=abps[i]
        if cur_time>start_time:
            progress=1
            result=np.vstack((result,np.array([cur_time,cur_abp,cur_icp])))
        if cur_time>end_time:
            progress=0
            print("Write")
            result_df=pd.DataFrame(result,columns=["DateTime","ABP","ICP"])
            result_df.to_csv(result_path+"/"+str(filename)+"_"+str(time_idx)+".csv",index=False)
            result=np.empty((0,3))
            time_idx=time_idx+1
            if time_idx>=max_time:
                break
    if progress==1:
        print("Write")
        result_df=pd.DataFrame(result,columns=["DateTime","ABP","ICP"])
        result_df.to_csv(result_path+"/"+str(filename)+"_"+str(time_idx)+".csv",index=False)
def save_pred_time(pred,timelapse,env):
    cut_range=env.get_config("test","cut_range",type="int")

    pred_times=np.empty((0,2))
    pred_progress=0;
    pred_start=0; pred_end=0;
    total_len=len(pred)
    for i in range(total_len):
        # pred
        if (pred[i]==1) and (pred_progress==0): # prediction segment start
            pred_progress=1
            if i-cut_range<0:
                pred_start=0
            else:
                pred_start=i-cut_range
        elif (pred[i]==0) and (pred_progress==1): # prediction segment end
            pred_progress=0
            if i+cut_range>total_len:
                pred_end=total_len
            else:
                pred_end=i+cut_range
           
            time_row=np.array([timelapse[pred_start],timelapse[pred_end-1]])
            pred_times=np.vstack((pred_times,time_row))
        else:
            continue
    return pred_times
def draw_graph(data,true,pred,fileidx,env,idx_group=[]):
    figure_x=env.get_config("test","figure_x",type="int")
    figure_y=env.get_config("test","figure_y",type="int")
    figure_range=env.get_config("test","figure_range",type="int")

    graph_path=env.get_config("path","graph_path")

    if len(idx_group)==0:
        idx_group=[num for num in range(len(data))]

    figure_num=0
    for i in range(len(idx_group)):
        cur_group=idx_group[i]
        if i%figure_range==0:
            if figure_num>0:
                plt.savefig(graph_path+"/"+str(fileidx)+"_"+str(figure_num)+'.png')
            figure_num=figure_num+1
            plt.figure(figure_num,figsize=(figure_x,figure_y*figure_range))
        plt.subplot(figure_range*100+10+((i%figure_range)+1))
        timelapse=data[0][cur_group]
        icp=data[1][cur_group]
        cur_true=np.multiply(true[cur_group],max(icp))
        cur_pred=np.multiply(pred[cur_group],max(icp))
        
        plt.plot(timelapse,icp,'b',label="ICP")
        plt.plot(timelapse,cur_true.reshape(-1),'g',label="true")
        plt.plot(timelapse,cur_pred.reshape(-1),'r--',label="prediction")
        # plt.grid()
        plt.legend()
    plt.savefig(graph_path+"/"+str(fileidx)+"_"+str(figure_num)+'.png')
    plt.clf()
    return figure_num

def confusion_result(confusion,env):
    tp=confusion[0]; tn=confusion[1]; fp=confusion[2]; fn=confusion[3]
    tpr=tp/(tp+fn)
    tnr=tn/(tn+fp)
    ppv=tp/(tp+fp)
    npv=tn/(fn+tn)
    acc=(tp+tn)/(tp+tn+fp+fn)
    netpred=(tpr+tnr)/2
    
    print("TP : "+str(tp))
    print("TN : "+str(tn))
    print("FP : "+str(fp))
    print("FN : "+str(fn))
    print("Sensitivity : "+str(tpr))
    print("Specificity : "+str(tnr))
    print("Net Prediction : "+str(netpred))
    print("PPV : "+str(ppv))
    print("NPV : "+str(npv))
    print("Accuracy : "+str(acc))
    
    save_confusion(np.array([tp,tn,fp,fn,tpr,tnr,netpred,ppv,npv,acc]),env)

def segment_confusion_result(confusion,env):
    tp=confusion[0]; tn=confusion[1]; fp=confusion[2]; fn=confusion[3]
    ppv=tp/(tp+fp)
    acc=(tp+tn)/(tp+tn+fp+fn)
    
    print("TP : "+str(tp))
    print("FP : "+str(fp))
    print("FN : "+str(fn))
    print("PPV : "+str(ppv))
    print("Accuracy : "+str(acc))
    
    save_confusion(np.array([tp,tn,fp,fn,ppv,acc]),env)

def save_confusion(confusion,env):
    new_row=[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    new_row.extend(confusion)
    result_path=env.get_config("path","result_path")
    with open(result_path+"/result.csv",'a') as f:
        writer=csv.writer(f)
        writer.writerow(new_row)

def get_confusion(true, pred):
    if len(true)!=len(pred):
        print("Wrong Input")
        return
    else:
        tp=0; tn=0; fp=0; fn=0
        for i in range(len(true)):
            if true[i]==pred[i]:
                if true[i]==0:
                    tn=tn+1
                else:
                    tp=tp+1
            else:
                if pred[i]==0:
                    fn=fn+1
                else:
                    fp=fp+1    
        return np.array([tp,tn,fp,fn])

def get_segment_confusion(true,pred):
  if len(true)!=len(pred):
        print("Wrong Input")
        return
  else:
    tp=0; tn=0; fp=0; fn=0
    pred_progress=0; true_progress=0
    pred_start=0; pred_end=0; true_start=0; true_end=0
    for i in range(len(true)):
        # pred
        if (pred[i]==1) and (pred_progress==0): # prediction segment start
            pred_progress=1
            pred_start=i
        elif (pred[i]==0) and (pred_progress==1): # prediction segment end
            pred_progress=0
            pred_end=i+1
            # Decide TP of FP
            if any(true[pred_start:pred_end]):
                tp=tp+1
            else:
                fp=fp+1
        # true
        if (true[i]==1) and (true_progress==0): # label segment start
            true_progress=1
            true_start=i
        elif (true[i]==0) and (true_progress==1): # label segment end
            true_progress=0
            true_end=i+1
            # Check FN
            if not any(pred[true_start:true_end]):
                fn=fn+1
    return np.array([tp,tn,fp,fn])