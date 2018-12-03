from __future__ import print_function

import torch
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile
import torchvision.transforms as trans
from PIL import Image

#from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import model
import net_sphere

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold



parser = argparse.ArgumentParser(description='PyTorch  lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--lfw', default='/root/NewDisk/daxing2/Evaluation/data/lfw112x96/', type=str)
parser.add_argument('--model','-m', default='./work_space/model/model_mobilefacenet.pth', type=str)#sphere20a_20171020.pth  model_mobilefacenet.pth
args = parser.parse_args()

predicts=[]
#net = getattr(net_sphere,args.net)()
net = model.MobileFaceNet(512)

net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True
test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
#zfile = zipfile.ZipFile(args.lfw)
lfw_dir = args.lfw
landmark = {}
with open('data/lfw_landmark.txt') as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [int(k) for k in l[1:]]

with open('data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]

for i in range(6000):
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
    #img11 = cv2.resize(cv2.imread(os.path.join(lfw_dir,name1)),(112,112))
    #img22 = cv2.resize(cv2.imread(os.path.join(lfw_dir, name2)), (112, 112))
    img1 = cv2.imread(os.path.join(lfw_dir,name1))#alignment(cv2.imread(os.path.join(lfw_dir,name1)),landmark[name1])#cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1)
    img2 = cv2.imread(os.path.join(lfw_dir,name2))#alignment(cv2.imread(os.path.join(lfw_dir,name2)),landmark[name2])#cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1)
    img1 = cv2.resize(img1,(112,112))
    img2 = cv2.resize(img2,(112,112))

    imglist = [img1,img2]#[img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    for j in range(len(imglist)):
        #imglist[j] = imglist[j].transpose(2, 0, 1)
       # imglist[j] = imglist[j].transpose(2, 0, 1).reshape((1,3,112,96))
       # imglist[j] = (imglist[j]-127.5)/128.0
        ### insight face
        imglist[j] = test_transform(imglist[j])#(imglist[i]-127.5)/128.0
        imglist[j] = imglist[j].reshape((1,3,112,112))

    img = torch.cat([imglist[0],imglist[1]],0).cuda()#np.vstack(imglist)

    #img = np.vstack(imglist)
    #img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    with torch.no_grad():
        output = net(img)
    f = output.data
    f1,f2 = f[0],f[1]
    #diff = np.sum(np.square((f1-f2),1))
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))
    if i %100 == 0:
        print ('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


accuracy = []
thd = []
folds = KFold(n=6000, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(list(map(lambda line:line.strip('\n').split(), predicts)))
for idx, (train, test) in enumerate(folds):
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f} '.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
