from cgi import print_arguments
import os
import torch
import time
import random
from torch import nn,square
import torch.nn as nn
import pandas as pd
import numpy as np
from torch import Tensor
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from conv import model
from get_dataset import gets_data
from torch.optim import lr_scheduler
#gpu setting
# device = "cuda" if torch.cuda.is_available() else "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "3"  #（代表仅使用第3，4号GPU）


# Initialize the Weights、bias
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02) 
#         # m.weight.data是卷积核参数, m.bias.data是偏置项参数
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)



#define optimizer
#optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)

#define train
class Train():
    def __init__(self,nsub):
        super(Train, self).__init__()
        self.batch_size = 32
        self.n_epochs = 150
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.99
        self.nSub = nsub
        self.model = model().cuda()
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [40,80], 0.1)
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.log_write = open("/home/gncui/EEG-GAN/results/log_subject_1000%d.txt" % self.nSub, "w")
    def train(self):
        train_data, train_label, train_label_id, test_data, test_label, test_label_id = gets_data(self.nSub)
        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label - 1)
        train_label_id = torch.from_numpy(train_label_id - 1)

        dataset = torch.utils.data.TensorDataset(train_data, train_label, train_label_id)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size = self.batch_size, shuffle=True)


        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_label_id = torch.from_numpy(test_label_id - 1)

        test_dataset = torch.utils.data.TensorDataset(test_data, test_label, test_label_id)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = self.batch_size, shuffle=False)



        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        test_label_id = Variable(test_label_id.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0


        # some better optimization strategy is worthy to explore. Sometimes terrible over-fitting.


        for e in range(self.n_epochs):
            print(f"Epoch {e+1}\n-------------------------------")
            in_epoch = time.time()
            #training
            self.scheduler.step()
            size_train = len(dataloader.dataset)
            self.model.train()
            train_acc = 0
            train_acc_id = 0
            acc = 0
            acc_id = 0
            loss_TEST = 0
            loss_TEST_id = 0
            for batch, (img, label, label_id) in enumerate(dataloader):
                #pdb.set_trace()

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                #label_id = Variable(label_id.cuda().type(self.LongTensor))
 
                outputs = self.model(img)

                # MI classification loss
                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # outputs_id = self.model.grl_forward(img)
                # # ID classification loss
                # loss_id = self.criterion_cls(outputs_id, label_id)
                # self.optimizer.zero_grad()
                # loss_id.backward()
                # self.optimizer.step()

                train_pred = torch.max(outputs, 1)[1]
                #train_pred_id = torch.max(outputs_id, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) + train_acc
                #train_acc_id = float((train_pred_id == label_id).cpu().numpy().astype(int).sum()) + train_acc_id
                if batch % 100 == 0:
                    print('Train loss:', loss.detach().cpu().numpy(),)#'Train_id loss:', loss_id.detach().cpu().numpy(),)
            train_acc = float(train_acc/size_train)
            #train_acc_id = float(train_acc_id/size_train)
            out_epoch = time.time()

            # test
            size_test = len(test_dataloader.dataset)
            num_batches = len(test_dataloader)
            self.model.eval()
            for m, (img, label, label_id) in enumerate(test_dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                #label_id = Variable(label_id.cuda().type(self.LongTensor))
                with torch.no_grad():
                    Cls = self.model(img)
                    #Cls_id =  self.model.grl_forward(img)
                loss_test = self.criterion_cls(Cls, label)
                #loss_test_id = self.criterion_cls(Cls_id, label_id)
                loss_TEST = loss_test + loss_TEST
                #loss_TEST_id = loss_test_id + loss_TEST_id
                y_pred = torch.max(Cls, 1)[1]
                #y_pred_id = torch.max(Cls_id, 1)[1]
                acc = float((y_pred == label).cpu().numpy().astype(int).sum()) + acc 
                #acc_id = float((y_pred_id == label_id).cpu().numpy().astype(int).sum()) + acc_id
            acc = float(acc/size_test)
            #acc_id = float(acc_id/size_test)
            loss_TEST = loss_TEST/num_batches
            #loss_TEST_id = loss_TEST_id/num_batches
            print(
                      '  Train loss:', loss.detach().cpu().numpy(),
                      #'  Train_id loss:', loss_id.detach().cpu().numpy(),
                      '  Test loss:', loss_TEST.detach().cpu().numpy(),
                      #'  Test_id loss:', loss_TEST_id.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      #'  Train_id accuracy:', train_acc_id,
                      '  Test accuracy is:', acc)
                     # '  Test_id accuracy is:', acc_id)
            self.log_write.write(str(e) + "    " + str(acc) + "\n")
            num = num + 1
            averAcc = averAcc + acc
            if acc > bestAcc:
                bestAcc = acc
                Y_true = test_label
                Y_pred = y_pred

        torch.save(self.model.state_dict(), 'model_1000.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred





def main():
    best = 0
    aver = 0
   # root = '/home/gncui/EEG-Transformer-main/MI_dataset/'
    result_write = open('/home/gncui/EEG-GAN/results/sub_result_1000.txt', "w")

    for i in range(9):
        seed_n = np.random.randint(500)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('Subject %d' % (i+1))
        training = Train(i + 1)
        bestAcc, averAcc, Y_true, Y_pred = training.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('**Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
        # plot_confusion_matrix(Y_true, Y_pred, i+1)
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    main()
