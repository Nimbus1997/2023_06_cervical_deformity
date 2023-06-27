# Renset50 - pretrained but update all parameters
# edit 
# 1) best epoch - save pth (2023.06.20)
# 2) weighted loss - to overcome the biased data distribution (2023.06.20)

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


# 데이터 폴더 경로 - Setting!!! =========================================
data_dir = "/root/jieunoh/cervical_deformity/data"
result_folder = "/root/jieunoh/cervical_deformity/result/vgg13/"
result_name = "0620_vgg13_bn_allclass"
result_folder = result_folder + result_name
# # 하이퍼파라미터 설정
num_epochs = 100
gpuid = 2
resizee =512
class_counts = [113, 119, 276, 240] # edit 2 class_labels 순서 ['Cancer', 'HSIL', 'LSIL', 'normal']
# [END] 데이터 폴더 경로 - Setting!!! =========================================


if not os.path.isdir(result_folder):
	os.makedirs(result_folder)
        
# GPU 사용 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:%d"%(gpuid))

# 학습 중 train accuracy와 validation accuracy를 그래프로 저장하기 위한 함수
def save_accuracy_graph(train_acc_list, val_acc_list,class_acc_dict, fold, best_epoch):
    epochs = len(train_acc_list)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    # Class-wise accuracy
    for class_name, acc_list in class_acc_dict.items():
        plt.plot(range(1, epochs+1), acc_list, label=f'Class {class_name} Accuracy')
    title = result_name+", best epoch = "+str(best_epoch)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(result_folder+'/accuracy_graph_%d.png'%(fold))
    plt.close()


# # 데이터 전처리
pretransform = transforms.Compose([transforms.Resize((resizee,resizee)), transforms.ToTensor()])
# 데이터셋 불러오기
dataset = datasets.ImageFolder(root=data_dir, transform=pretransform)
class_labels = dataset.classes
total_samples = sum(class_counts) 
class_weights = [total_samples / counts for counts in class_counts]
class_weights = torch.Tensor(class_weights).to(device)

# # 3-fold cross validation을 위한 인덱스 생성
kfold = KFold(n_splits=3, shuffle = True)

# # 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss(weight=class_weights) #  edit 2


# prameter 수
num_classes = len(dataset.classes)
model = models.vgg13_bn(pretrained = True)
model.classifier[6] = nn.Linear(4096, num_classes)
total_params = sum(p.numel() for p in model.parameters())
print("total_params:", total_params)

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    # for saving best model 
    val_accuracy_best = 0 # edit 1
    best_epoch = 0
        
    # 모델 정의 및 pretrained 가중치 로드
    model = models.vgg13_bn(pretrained = True)
    model.classifier[6] = nn.Linear(4096, num_classes) # 마지막 레이어를 num_classes에 맞게 수정
    model = model.to(device)

    # 학습 중 train accuracy와 validation accuracy를 기록하기 위한 리스트
    train_acc_list = []
    val_acc_list = []
    class_acc_dict_val = {class_label: [] for class_label in class_labels}

    print(fold, len(train_idx), len(val_idx))
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

        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        # 클래스별 정확도 계산
        class_correct = list(0.0 for _ in range(num_classes))
        class_total = list(0.0 for _ in range(num_classes))

        with torch.no_grad():
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
            val_loss = val_loss / len(val_idx)
            val_accuracy = val_correct / len(val_idx)
            if val_accuracy_best < val_accuracy:
                val_accuracy_best = val_accuracy
                torch.save(model.state_dict(), result_folder+f'/model_fold{fold}_best.pth')
                best_epoch = epoch
                print("best_fold",str(fold))


            # 클래스별 정확도 출력
            for i in range(num_classes):
                class_acc = 100 * class_correct[i] / class_total[i]
                print(class_labels[i], str(class_acc))
                class_acc_dict_val[class_labels[i]].append(class_acc)

    # 결과 출력
        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        train_acc100 = 100.0 * train_accuracy
        val_acc100 = 100.0 * val_accuracy
        train_acc_list.append(train_acc100)
        val_acc_list.append(val_acc100)

        save_accuracy_graph(train_acc_list, val_acc_list, class_acc_dict_val, fold, best_epoch)
        torch.save(model.state_dict(), result_folder+f'/model_fold{fold}.pth')

print("best epoch: ", best_epoch)