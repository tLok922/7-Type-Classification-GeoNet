#!/home/bsft21/tinloklee2/miniconda3/envs/depth/bin/python

# IMPORTANT: Use LF as EOF Sequence
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from dataloaderNetM import get_loader
from modelNetM import EncoderNet, DecoderNet, ClassNet, EPELoss
from tqdm import tqdm

parser = argparse.ArgumentParser(description='GeoNetM')
parser.add_argument('--epochs', type=int, default=5, metavar='N')
parser.add_argument('--reg', type=float, default=0.1, metavar='REG')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
parser.add_argument('--data_num', type=int, default=21900, metavar='N') #depreciated
parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument("--datasetdir", type=str, default='/public/tinloklee2/distorted_place365')
parser.add_argument("--distortion_types", type=list, default=['barrel','pincushion','shear','rotation','projective','wave','none'])
parser.add_argument("--pretrained_en", type=str, default='/home/bsft21/tinloklee2/Depth-Anything/GeoProj/model_en.pkl')
parser.add_argument("--pretrained_de", type=str, default='/home/bsft21/tinloklee2/Depth-Anything/GeoProj/model_de.pkl')
parser.add_argument("--pretrained_class", type=str, default=None)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader = get_loader(distortedImgDir = f'{args.datasetdir}/train/image',
                  flowDir       = f'{args.datasetdir}/train/flow', 
                  batch_size = args.batch_size,
                  distortion_type = args.distortion_types,
                  data_num = args.data_num)

model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])
model_class = ClassNet()
criterion = EPELoss()
criterion_clas = nn.CrossEntropyLoss()

if args.pretrained_en: 
    model_en.load_state_dict(torch.load(args.pretrained_en))
if args.pretrained_de:
    model_de.load_state_dict(torch.load(args.pretrained_de))
if args.pretrained_class:
    model_de.load_state_dict(torch.load(args.pretrained_class))

print('dataset type:',args.distortion_types)
# print('dataset number:',args.data_num)
print('batch size:', args.batch_size)
print('epochs:', args.epochs)
print('lr:', args.lr)
print('reg:', args.reg)
print('train_loader',len(train_loader), 'train_num', args.batch_size*len(train_loader))
# print('val_loader', len(val_loader),   'val_num', args.batch_size*len(val_loader))
print(model_en, model_de, model_class, criterion)

if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    model_en = nn.DataParallel(model_en)
    model_de = nn.DataParallel(model_de)
    model_class = nn.DataParallel(model_class)

model_en = model_en.to(device)
model_de = model_de.to(device)
model_class = model_class.to(device)
criterion = criterion.to(device)
criterion_clas = criterion_clas.to(device)

reg = args.reg
lr = args.lr
optimizer = torch.optim.Adam(list(model_en.parameters()) + list(model_de.parameters()) + list(model_class.parameters()), lr=lr)

model_en.train()
model_de.train()
model_class.train()

for epoch in range(args.epochs):
    for i, (disimgs, disx, disy, labels) in enumerate(train_loader):
        disimgs = disimgs.to(device)
        disx = disx.to(device)
        disy = disy.to(device)
        labels = labels.to(device)
        
        # disimgs = Variable(disimgs)
        # labels_x = Variable(disx)
        # labels_y = Variable(disy)
        # labels_clas = Variable(labels)
        flow_truth = torch.cat([disx, disy], dim=1)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        
        middle = model_en(disimgs)
        flow_output = model_de(middle)
        clas = model_class(middle)
        
        loss1 = criterion(flow_output, flow_truth)
        loss2 = criterion_clas(clas, labels)*reg
       
        loss = loss1 + loss2
            
        loss.backward()
        optimizer.step()
        
        # print log
        print("Epoch [%d], Iter [%d], Loss: %.4f, Loss1: %.4f, Loss2: %.4f" %(epoch + 1, i + 1, loss.data[0], loss1.data[0], loss2.data[0]))
     
    torch.save(model_en.state_dict(), f'7-type_model_en_{epoch+1}.pkl') 
    torch.save(model_de.state_dict(), f'7-type_model_de_{epoch+1}.pkl') 
    torch.save(model_class.state_dict(), f'7-type_model_class_{epoch+1}.pkl') 