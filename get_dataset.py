import scipy.io
import numpy as np
from torch.utils.data import DataLoader, Dataset

def gets_data(sub):
    root = '/home/gncui/EEG-Transformer-main/MI_transformer_z/'  # the path of data
    total_data = scipy.io.loadmat(root + 'all_1000.mat' )
    all_data = total_data['data']
    all_label = total_data['label']
    all_label_id = total_data['label1']
    all_data = np.transpose(all_data, (2, 1, 0))
    all_data = np.expand_dims(all_data, axis=1)
    all_label = np.transpose(all_label)
    all_label_id = np.transpose(all_label_id)
    allData = all_data
    allLabel = all_label[0]
    allLabel_id = all_label_id[0]

    # datalist = []
    # datalabellist = []
    # datalabelidlist = []

    #按照对象调整数据集顺序
    # for i in range(9):
    #     data1 = allData[i*288:(i+1)*288]
    #     data2 = allData[(i+9)*288:(i+10)*288]
    #     datalist.append(np.concatenate((data1,data2),0))
    #     label1 = allLabel[i*288:(i+1)*288]
    #     label2 = allLabel[(i+9)*288:(i+10)*288] 
    #     datalabellist.append(np.concatenate((label1,label2),0))
    #     labelid1 = allLabel_id[i*288:(i+1)*288]
    #     labelid2 = allLabel_id[(i+9)*288:(i+10)*288] 
    #     datalabelidlist.append(np.concatenate((labelid1,labelid2),0))
    # allData = np.concatenate(datalist,0)
    # allLabel = np.concatenate(datalabellist,0)
    # allLabel_id = np.concatenate(datalabelidlist,0)
        #which id(i)
        # train_data、test_data.
    train_data_1 = allData[:(sub-1)*576]
    train_data_2 = allData[sub*576:]
    train_data = np.concatenate((train_data_1, train_data_2), 0)
    test_data = allData[(sub-1)*576:sub*576]
        # train_label,test_label
    train_label_1 = allLabel[:(sub-1)*576]
    train_label_2 = allLabel[sub*576:]
    train_label = np.concatenate((train_label_1, train_label_2), 0)
    test_label = allLabel[(sub-1)*576:sub*576]
        # train_label_id,test_label_id
    train_label_id_1 = allLabel_id[:(sub-1)*576]
    train_label_id_2 = allLabel_id[sub*576:]
    train_label_id = np.concatenate((train_label_id_1, train_label_id_2), 0)
    test_label_id = allLabel_id[(sub-1)*576:sub*576]
        #Data augment
    train_datalist = []
    train_labellist = []
    train_label_id_list = []
    test_datalist = []
    test_labellist = []
    test_label_id_list = []
    for i in range(20):
        train_data1 = train_data[:,:,:,50*(i+1):1000]
        train_data2 = train_data[:,:,:,0:50*(i+1)]
        train_datalist.append(np.concatenate((train_data1,train_data2),3))
        test_data1 = test_data[:,:,:,50*(i+1):1000]
        test_data2 = test_data[:,:,:,0:50*(i+1)]
        test_datalist.append(np.concatenate((test_data1,test_data2),3))
    train_data = np.concatenate(train_datalist,0)
    test_data = np.concatenate(test_datalist,0)
    for i in range(20):
        train_labellist = np.concatenate((train_labellist,train_label),0)
        test_labellist = np.concatenate((test_labellist,test_label),0)
        train_label_id_list = np.concatenate((train_label_id_list,train_label_id),0)
        test_label_id_list = np.concatenate((test_label_id_list,test_label_id),0)
    train_label = train_labellist
    test_label = test_labellist
    train_label_id = train_label_id_list
    test_label_id = test_label_id_list
        #shuffle train_data
    all_shuff_num = np.random.permutation(len(train_data))
    train_data = train_data[all_shuff_num]
    train_label = train_label[all_shuff_num]
    train_label_id = train_label_id[all_shuff_num]
    return train_data, train_label, train_label_id, test_data, test_label, test_label_id
