#!/home/bsft21/tinloklee2/miniconda3/envs/depth/bin/python

# IMPORTANT: Use LF as EOF Sequence
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from dataloaderNetM import get_loader
from modelNetM import EncoderNet, DecoderNet, ClassNet, EPELoss
from tqdm import tqdm
import os
from torchvision import transforms
import numpy as np

parser = argparse.ArgumentParser(description='GeoNetM')
# parser.add_argument('--epochs', type=int, default=5, metavar='N')
parser.add_argument('--reg', type=float, default=0.1, metavar='REG')
# parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
parser.add_argument('--data_num', type=int, default=21900, metavar='N')
parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument("--dataset_dir", type=str, default='/public/tinloklee2/distorted_place365')
parser.add_argument("--distortion_types", type=list, default=['barrel','pincushion','shear','rotation','projective','wave','none'])
parser.add_argument("--pretrained_en", type=str, default='/home/bsft21/tinloklee2/Depth-Anything/GeoProj/7-type_model_en_5.pkl')
parser.add_argument("--pretrained_de", type=str, default='/home/bsft21/tinloklee2/Depth-Anything/GeoProj/7-type_model_de_5.pkl')
parser.add_argument("--pretrained_class", type=str, default='/home/bsft21/tinloklee2/Depth-Anything/GeoProj/7-type_model_class_5.pkl')
args = parser.parse_args()

testImgPath = os.path.join(args.dataset_dir,'test/image')
saveFlowPath = os.path.join(args.dataset_dir,'test/flow')

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_loader = get_loader(distortedImgDir = testImgPath,
                  flowDir       = saveFlowPath, 
                  batch_size = args.batch_size,
                  distortion_type = args.distortion_types,
                  data_num = args.data_num)

model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])
model_class = ClassNet()
# criterion = EPELoss()
# criterion_clas = nn.CrossEntropyLoss()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_en = nn.DataParallel(model_en)
    model_de = nn.DataParallel(model_de)
    # model_class = nn.DataParallel(model_class)

model_en.load_state_dict(torch.load(args.pretrained_en))
model_de.load_state_dict(torch.load(args.pretrained_de))
# model_class.load_state_dict(torch.load(args.pretrained_class))

model_en = model_en.to(device)
model_de = model_de.to(device)
model_class = model_class.to(device)
# criterion = criterion.to(device)
# criterion_clas = criterion_clas.to(device)

model_en.eval()
model_de.eval()
model_class.eval()

n_classes = len(args.distortion_types)
total_loss = 0.0
n_samples= 0
loss_by_class = [0.0]*n_classes
count_by_class = [0]*n_classes
# confusion_matrix = [[0]*n_classes for _ in range(n_classes)]

for i, (disimgs, disx, disy, labels) in enumerate(test_loader):
    disimgs = disimgs.to(device)
    disx = disx.to(device)
    disy = disy.to(device)
    labels = labels.to(device)
    
    # disimgs = Variable(disimgs)
    # labels_x = Variable(disx)
    # labels_y = Variable(disy)
    # labels_clas = Variable(labels)
    flow_truth = torch.cat([disx, disy], dim=1)
        
    middle = model_en(disimgs)
    flow_output = model_de(middle)
    clas = model_class(middle)
        
    # loss1 = criterion(flow_output, flow_truth)
    # loss2 = criterion_clas(clas, labels)*reg
    # loss = loss1 #+ loss2
    
    # EPELoss
    epe = torch.norm(flow_output - flow_truth + 1e-16, p=2, dim=1)
    per_sample_loss = epe.view(epe.size(0),-1).mean(dim=1)
    loss = per_sample_loss.mean()
    
    # useful info
    # https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1
    # https://stackoverflow.com/questions/42479902/what-does-view-do-in-pytorch

    total_loss += per_sample_loss.sum().item()
    n_samples += per_sample_loss.size(0)

    _, preds = torch.max(clas.data, 1)
    for loss_val, true_cls, pred_cls in zip(per_sample_loss, labels, preds):
        class_idx = true_cls.item()
        loss_by_class[class_idx]+=loss_val.item()
        count_by_class[class_idx]+=1
        # confusion_matrix[class_idx][pred_cls.item()]+=1 
    print(loss_by_class)
    # print(confusion_matrix)



avg_loss = total_loss/n_samples
avg_loss_by_class = [(loss_by_class[class_idx]/count_by_class[class_idx]) if count_by_class[class_idx] > 0 else 0.0 for class_idx in range(n_classes)]

print(f"Validation Loss: {avg_loss:.4f}")
print()
print("Validation Loss by Distortion Type:")
for class_idx, distortion_type in enumerate(args.distortion_types):
    print(f"    {distortion_type:15s}: {avg_loss_by_class[class_idx]:.4f}")
print()