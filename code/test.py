# from dataloader_v2 import UniDataloader
from model_v0 import GCN_Model as Model
# from model_v0 import GCN_Pool_Model as Model
# from model_v0 import DNN as Model
import yaml
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import numpy as np
import torch
# import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
# from torchsummary import summary


with open("config_v0.yaml") as file:
    config = yaml.safe_load(file)

if config['DATASET']['NUM_CLASSES']==2:
    from dataloader_v5 import UniDataloader
else:
    from dataloader_v2 import UniDataloader

### Fix Random Seed ###
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(config["SETTING"]["RANDOM_SEED"])
torch.cuda.manual_seed(config["SETTING"]["RANDOM_SEED"])
torch.cuda.manual_seed_all(config["SETTING"]["RANDOM_SEED"])
np.random.seed(config["SETTING"]["RANDOM_SEED"])
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(config["SETTING"]["RANDOM_SEED"])
########################

dataloader_class = UniDataloader(config)
train_dataloader_list, val_dataloader_list = dataloader_class.get_dataloader("cv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_edge = dataloader_class.dataset_class.max_edge
print("Max edge: ", max_edge)
# max_edge = 160

# total_results = [0, 0, 0, 0] # acc / precision / recall / f1
total_results = [[], [], [], []] # acc / precision / recall / f1
print_epoch = 50

# max_edge = 1419
# model = Model(max_edge, 1, 512, config['DATASET']['NUM_CLASSES'], 0.5).to(device)

# print(summary(model, [(1,1419,1),(1,1419,1419)]))
# print(summary(model, [(1,160,1),(1,160,160)]))

# model = Model(config, 512).to(device)
# print(summary(model, (1,530,530)))
# exit()

for i in range(config["DATASET"]["NUM_FOLD"]):
    print("##### Fold {} #####".format(i+1))

    model = Model(max_edge, 1, 512, config['DATASET']['NUM_CLASSES'], 0.5).to(device)
    # model = Model(config, 512//2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["NET"]["LR"]/10, weight_decay=config["NET"]["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)
    loss_function = nn.NLLLoss()

    train_dataloader = train_dataloader_list[i]
    val_dataloader = val_dataloader_list[i]

    for j in range(config["TRAIN"]["MAX_EPOCHS"]):
        if (j+1)%print_epoch==0: print("Epoch ", j+1, end=': ')
        results = [0, 0, 0, 0] # TP / FP / FN / TN

        model.train()
        for A_train, X_train, y_train in train_dataloader:
            A_train, X_train, y_train = A_train.cuda(), X_train.cuda(), y_train.cuda()
            pred = model(X_train, A_train)
            # pred = model(A_train.squeeze())
            loss = loss_function(pred, y_train)

            # l1_reg = torch.tensor(0.0).to(device)
            # for name, param in model.named_parameters():
            #     if name == "mlp.0.weight": # fc1.weight
            #         l1_reg += torch.norm(param, p=1)
            # loss += config["NET"]["L1"] * l1_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (j+1)%print_epoch == 0: print("Loss: {0:.6f}".format(loss.item()), end=' | ')
        else:
            print("Loss: {0:.6f}".format(loss.item()), end=' | ')


        if (j+1)%print_epoch==0:
            print("lr: ", optimizer.param_groups[0]['lr'])
            wrong = 0
            correct_sample = [[],[],[],[]]
            for A_val, X_val, y_val in val_dataloader:
                A_val, X_val, y_val = A_val.cuda(), X_val.cuda(), y_val.cuda()
                # with torch.no_grad():
                model.eval()
            
                prediction = model(X_val, A_val)
                # prediction = model(A_val)
                pred = torch.argmax(prediction, dim=1)

                for yhat, ypred in zip(y_val, pred):
                    if yhat==ypred:
                        if yhat == 0: results[0] += 1
                        elif yhat == 1: results[1] += 1
                        elif yhat == 2: results[2] += 1
                        else: results[3] += 1
                    else:
                        wrong += 1
            
            print(results)
            right = sum(results)

            acc = 100 * right / (right+wrong)
            if i==0 or i==1 or i==2: acc_CN = results[1] / (169) # 1
            else: acc_CN = results[1] / (168) # 1
            acc_EMCI = results[2] / (98) # 2
            acc_LMCI = results[3] / (50) # 3
            acc_AD = results[0] / (48) # 0

            prec, rec, f, _ = precision_recall_fscore_support(y_val.cpu(), pred.cpu(), average='macro')
            prec, rec, f = 100*prec, 100*rec, 100*f


            print("acc: {}, [CN: {}, EMCI: {}, LMCI: {}, AD: {}]".format(acc,acc_CN,acc_EMCI,acc_LMCI,acc_AD))
            print("Precision: {} | Recall: {} | F1: {}".format(prec, rec, f))

        # if j+1 == config["TRAIN"]["MAX_EPOCHS"]:
        if (j+1)%print_epoch==0:
            total_results[0].append(acc)
            total_results[1].append(prec)
            total_results[2].append(rec)
            total_results[3].append(f)

            #Grad-CAM
            for yhat, ypred, idx in zip(y_val, pred, range(right+wrong)):
                if yhat==ypred: correct_sample[yhat].append(idx)

            # for indx, label in zip(correct_sample,['AD','CN','EMCI','LMCI']):
            #     ix = indx[5]
            #     prediction[ix][pred[ix]].backward(retain_graph=True)
            #     gradients = model.get_activations_gradient()
            #     pooled_grad = torch.mean(gradients, dim=[0,2]).squeeze() # 512
            #     pooled_grad = torch.maximum(pooled_grad, torch.zeros_like(pooled_grad))

            #     activations = model.get_activations(X_val[ix], A_val[ix]).detach()
            #     for i in range(max_edge): activations[0,i,:] *= pooled_grad

            #     heatmap = torch.mean(activations, dim=2).squeeze()
            #     # if torch.max(heatmap)<=0: heatmap += (torch.abs(torch.min(heatmap))/2)
            #     # heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap)) # ReLU
            #     heatmap /= torch.max(heatmap) # Normalize
            #     print(label,end=': ')
            #     print(torch.topk(heatmap,10))

            sig_edges = []
            for indx, label in zip(correct_sample,['AD','CN','EMCI','LMCI']):
                # if label!='CN': continue
                heatmap_weights = []
                heatmap_edges = []
                for ix in indx:
                    prediction[ix][pred[ix]].backward(retain_graph=True)
                    gradients = model.get_activations_gradient()
                    pooled_grad = torch.mean(gradients, dim=[0,2]).squeeze() # 512
                    pooled_grad = torch.maximum(pooled_grad, torch.zeros_like(pooled_grad))

                    activations = model.get_activations(X_val[ix], A_val[ix]).detach()
                    for i in range(max_edge): activations[0,i,:] *= pooled_grad

                    heatmap = torch.mean(activations, dim=2).squeeze()
                    heatmap /= torch.max(heatmap) # Normalize
                    w, e = torch.topk(heatmap, 10)
                    heatmap_edges.append(e)
                    heatmap_weights.append(w)
                    # print(label,end=': ')
                    # print(torch.topk(heatmap,10))

                    print("grad: ",gradients.shape)
                    print("pooled: ",pooled_grad.shape)
                    print("activ: ",activations.shape)
                    print("heat: ",heatmap.shape)
                    print("X:",X_val[ix].shape)



            # print(heatmap_edges)
            # print(heatmap_weights)
                temp = torch.stack(heatmap_edges,0)
                heatmap_unique_edges, unique_index = torch.unique(temp, return_inverse=True)
                sig_edges.append(heatmap_unique_edges)
                tp_w = torch.stack(heatmap_weights,0)
                aaaa = []
                for i in range(heatmap_unique_edges.shape[0]):
                    if torch.sum(tp_w.isnan())>0:
                        aaaa.append(torch.zeros(1))
                        continue
                    edge_w = torch.Tensor(np.ma.masked_equal(unique_index.cpu().numpy(),i).mask).cuda() * tp_w
                    ed = torch.sum(edge_w)
                    ed = torch.sum(ed)
                    
                    aaaa.append(ed / torch.count_nonzero(edge_w))
                    # aaaa.append(torch.Tensor(np.ma.masked_equal(unique_index.cpu().numpy(),i)).cuda() * tp_w)
                aaaa = torch.stack(aaaa,0)
                print(aaaa)
                print(aaaa.shape)
                # exit()


                print(heatmap_unique_edges)
                print(heatmap_unique_edges.shape)

                exit()

            

            
        scheduler.step()


# print("#########")
# for i in range(4): print(total_results[i]/config["DATASET"]["NUM_FOLD"])

print("#########")
total_results = np.array(total_results, dtype=object)
print("total_results\n",total_results)

print("Mean")
for i in range(4): 
    print(np.mean(np.array(total_results[i])))

print("Std")
for i in range(4): 
    print(np.std(np.array(total_results[i])))


                    
        #             for yhat, ypred in zip(y_val, pred):
        #                 if yhat==ypred:
        #                     if yhat == 0: results[3] += 1
        #                     else: results[0] += 1
        #                 else:
        #                     if yhat == 0: results[1] += 1
        #                     else: results[2] += 1

        #     print(results)
        #     acc = 100*(results[0]+results[3])/sum(results)
        #     precision = 100*results[0]/(results[0]+results[1])
        #     recall = 100*results[0]/(results[0]+results[2])
        #     f1 = 2*precision*recall/(precision+recall)

        #     print("acc: ", acc)
        #     print("precision: ", precision)
        #     print("recall: ", recall)
        #     print("f1: ", f1)
        #     # print(optimizer.param_groups[0]["lr"])


        # if j+1 == config["TRAIN"]["MAX_EPOCHS"]:
        #     total_results[0].append(acc)
        #     total_results[1].append(precision)
        #     total_results[2].append(recall)
        #     total_results[3].append(f1)
    #     scheduler.step()

    # edge_weights = []
    # for name, param in model.named_parameters():
    #     if name == "mlp.0.weight":
    #         for i in range(max_edge):
    #             edge_weights.append(sum(map(sum, param[:,512*i:512*(i+1)].tolist())))
    # print("Top-10 edges:")
    # print(torch.topk(torch.tensor(edge_weights),10))


# print("#########")
# total_results = np.array(total_results, dtype=object)
# print("total_results\n",total_results)

# print("Mean")
# for i in range(4): 
#     print(np.mean(np.array(total_results[i])))

# print("Std")
# for i in range(4): 
#     print(np.std(np.array(total_results[i])))


        # if j+1 == config["TRAIN"]["MAX_EPOCHS"]:
        #     total_results[0] += acc
        #     total_results[1] += precision
        #     total_results[2] += recall
        #     total_results[3] += f1


# print("#########")
# for i in range(4): print(total_results[i]/config["DATASET"]["NUM_FOLD"])