# Renset101 - pretrained but update all parameters
# edit 
# 1) best epoch - save pth (2023.06.20)
# 2) weighted loss - to overcome the biased data distribution (2023.06.20)
# 3) confusion matrix (2023.06.28)
# 4) random seed 고정 (2023.06.28)
# 5) loss plot color & line change (2023.06.28)
# 
# 실행방법
# 1) change 내용 변경
# 2) python 파일명.py


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import models
import torchvision
import torch
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pdb
from sklearn.metrics import confusion_matrix # edit 3
import seaborn as sns # edit 3
import numpy as np


# 데이터 폴더 경로 - Setting!!! =========================================
data_dir = "/root/jieunoh/cervical_deformity/data"
result_folder = "/root/jieunoh/cervical_deformity/result/resnet101/"
result_name = "resnet101_allclass_0628"
result_folder = result_folder + result_name
# # 하이퍼파라미터 설정
num_epochs = 150
gpuid = 0
resizee =512
class_counts = [113, 119, 276, 240] # edit 2 class_labels 순서 ['Cancer', 'HSIL', 'LSIL', 'normal']
seed = 42
# [END] 데이터 폴더 경로 - Setting!!! =========================================
if not os.path.isdir(result_folder):
	os.makedirs(result_folder)
if not os.path.isdir(os.path.join(result_folder,"confusion_matrices")):
	os.makedirs(os.path.join(result_folder,"confusion_matrices"))
        
# GPU 사용 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:%d"%(gpuid))



# def --------------------------------------------------------------------------

# 학습 중 train accuracy와 validation accuracy를 그래프로 저장하기 위한 함수
def save_accuracy_graph(train_acc_list, val_acc_list,class_acc_dict, fold, best_epoch):
    colorset = ["lightcoral", "goldenrod", "yellowgreen", "cadetblue"] # cancer (연 빨강) - HSIL (연 노랑) - LSIL () - normal 순
    i =0

    plt.plot(train_acc_list, label='Train Accuracy', color="darkslateblue", linewidth =2) # linewidth default=1.5
    plt.plot(val_acc_list, label='Validation Accuracy', color="firebrick", linewidth =2)
    # Class-wise accuracy
    for class_name, acc_list in class_acc_dict.items():
        plt.plot(acc_list, label=f'Class {class_name} Accuracy', color=colorset[i], linewidth=1)
        i+=1
    title = result_name+", best epoch = "+str(best_epoch)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(result_folder+'/accuracy_graph_%d.png'%(fold))
    plt.close()

# confusion matrix 저장 및 반환하기 위한 함수 
def confusion_matrix_save(labels_list, predicted_list, epoch, best, fold):
    # 이름 setting
    title = "Confusion Matrix, fold = %d, epoch = %d (now, best)"%(fold, epoch)
    if best:
        cm_save_path =os.path.join(result_folder,"0_f%d_confusion_matrix_best.png"%fold)
    else:
        cm_save_path =os.path.join(result_folder, "confusion_matrices", "f%d_confusion_matrix_%d.png"%(fold, epoch))

    
    confusion_mat = confusion_matrix(labels_list, predicted_list)
    # Define class labels
    class_labels = ['Cancer', 'HSIL', 'LSIL', 'Normal']
    # Create a figure and axes
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=True)
    # Set labels, title, and tick parameters
    ax.xaxis.set_label_position('top')  # Move x-axis labels to the top
    ax.xaxis.tick_top()  # Move x-axis ticks to the top
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(class_labels)
    ax.yaxis.set_ticklabels(class_labels)

    # Save the figure
    plt.savefig(cm_save_path)
    plt.close()


# code 시작 --------------------------------------------------------------------------
# # 데이터 전처리
pretransform = transforms.Compose([transforms.Resize((resizee,resizee)), transforms.ToTensor()])
# 데이터셋 불러오기
dataset = datasets.ImageFolder(root=data_dir, transform=pretransform)
class_labels = dataset.classes
total_samples = sum(class_counts) 
class_weights = [total_samples / counts for counts in class_counts]
class_weights = torch.Tensor(class_weights).to(device)

# # 3-fold cross validation을 위한 인덱스 생성
kfold = KFold(n_splits=3, shuffle = True, random_state=seed)

# # 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss(weight=class_weights) #  edit 2
# for saving best model 
val_accuracy_best = 0 # edit 1
best_epoch = 0

# prameter 수
num_classes = len(dataset.classes)
model = models.resnet101(pretrained = True)
model.fc = nn.Linear(2048, num_classes) 
total_params = sum(p.numel() for p in model.parameters())
print(">>> Total_params: %d <<<"%total_params)


# training & validation(for saving the best epoch) ----------------
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    # for saving best model 
    val_accuracy_best = 0 # edit 1
    best_epoch = 0
    # 모델 정의 및 pretrained 가중치 로드
    model = models.resnet101(pretrained = True) 
    model.fc = nn.Linear(2048, num_classes)  # 마지막 레이어를 num_classes에 맞게 수정
    model = model.to(device)

    # 학습 중 train accuracy와 validation accuracy를 기록하기 위한 리스트
    train_acc_list = []
    val_acc_list = []
    class_acc_dict_val = {class_label: [] for class_label in class_labels}

    print(">>> FOLD: %d, # of training data: %d, # of test data: %d <<<"%(fold, len(train_idx), len(val_idx)))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) # index 생성
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx) # index 생성
    
    # sampler를 이용한 DataLoader 정의
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler) # 해당하는 index 추출
    valloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=val_subsampler)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=1e-7)


    # 훈련 및 검증 반복
    for epoch in range(1, num_epochs + 1):
        # 훈련
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in trainloader:
            # print(images,labels)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # print(images,labels)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            train_loss += loss.item() * images.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss = train_loss / len(train_idx)
        train_accuracy = train_correct / len(train_idx)

        print(train_loss,train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        # 클래스별 정확도 계산
        class_correct = list(0.0 for _ in range(num_classes))
        class_total = list(0.0 for _ in range(num_classes))

        with torch.no_grad():
            predicted_list =[] # for confusion matrix (edit 3)
            labels_list =[] # for confusion matrix (edit 3)

            for images, labels in valloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels).squeeze()
                val_loss += loss.item() * images.size(0)
                val_correct += (predicted == labels).sum().item()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                predicted_list.extend(predicted.cpu().numpy()) # for confusion matrix (edit 3)
                labels_list.extend(labels.cpu().numpy()) # for confusion matrix (edit 3)
            val_loss = val_loss / len(val_idx)
            val_accuracy = val_correct / len(val_idx)

            # (edit 3) confusion matrix- save every 10 epoch ----
            if epoch%10 ==0:
                confusion_matrix_save(labels_list, predicted_list, epoch, False, fold= fold)

            # best val acc -----
            #   1) model parameter .pth save
            #   2) print "best"
            #   3) confusion matrix _ best version save
            if val_accuracy_best < val_accuracy:
                val_accuracy_best = val_accuracy
                # 1) model parameter .pth save
                torch.save(model.state_dict(), result_folder+f'/model_fold{fold}_best.pth') 
                # 2) print "best"
                best_epoch = epoch
                # 3) confusion matrix _ best version save
                confusion_matrix_save(labels_list, predicted_list, epoch, True, fold = fold)



    # 결과 출력
        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        train_acc100 = 100.0 * train_accuracy
        val_acc100 = 100.0 * val_accuracy
        train_acc_list.append(train_acc100)
        val_acc_list.append(val_acc100)
        # 클래스별 정확도 출력
        for i in range(num_classes):
            class_acc = 100 * class_correct[i] / class_total[i]
            print(class_labels[i], str(class_acc))
            class_acc_dict_val[class_labels[i]].append(class_acc)

        save_accuracy_graph(train_acc_list, val_acc_list, class_acc_dict_val, fold, best_epoch)
        torch.save(model.state_dict(), result_folder+f'/model_fold{fold}_last.pth')

