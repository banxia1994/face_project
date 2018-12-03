from __future__ import print_function

import torch
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random
import argparse
import numpy as np
import torchvision.transforms as trans
import pickle

from matlab_cp2tform import get_similarity_transform_for_cv2
from mtcnn_.src.align_trans import  warp_and_crop_face
import model
import net_sphere

from mtcnn_.src import detect_faces,show_bboxes
from PIL import Image

model_dir = './work_space/model/'
test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]

parser = argparse.ArgumentParser(description='eval similarity of two img or video')
parser.add_argument('--net','-n',default='mobileface',type = str,choices=['sphere,mobileface'])
parser.add_argument('--imgdir','-id',type=str,default='/root/NewDisk/daxing2/WW/InsightFace_Pytorch/data/test/',help='img dir or img floder with many pics') # test gallery
#parser.add_argument('--model','-m',default=model_dir+'sphere20a_20171020.pth',type=str)
# extra mean to extract feature
parser.add_argument('--phase','-p',default='eval',type=str,choices=['extra','eval'])
parser.add_argument('--cuda','-c',default=True)
parser.add_argument('--align',default=True)
parser.add_argument('--threshold','-th',default=0.32,type=float)
args = parser.parse_args()

print (args.phase)

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

def align(img,landmarks,size):
    facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, REFERENCE_FACIAL_POINTS, crop_size=size)
    return Image.fromarray(warped_face)

def extract_feature(net,img):
    with torch.no_grad():
        output = net(img)
    return output.data

if __name__ == '__main__':
    #args = args_().parse_args()
    imgs = {}
    final_imgs = {}
    feature = {}
    if os.path.isdir(args.imgdir):
        imglist = os.listdir(args.imgdir)
        for i in imglist:
            #img = cv2.imread(i)
            img = Image.open(os.path.join(args.imgdir,i))
            imgs[i.strip().split('/')[-1]] = img
    else:
        imgs[args.imgdir.strip().split('/')[0]] = Image.open(args.imgdir)

    if args.align:
        for id,item in imgs.items():
            # TODO: judge how many faces in img (default: only 1)
            bounding_boxes, landmarks = detect_faces(item)
            img = show_bboxes(item, bounding_boxes, landmarks)
            img.save('a.jpg')
            img = show_bboxes(img, bounding_boxes, landmarks)
            item = np.asarray(item)
            item = item[:,:,(2,1,0)] # change channle to BGR (PIL default is RBG)
            #img = alignment(item,landmarks[0])
            #img = align(item, landmarks,(112,112))

            #### test align
            #img = Image.fromarray(img.astype('uint8')).convert('RGB')
            #img.save('c.jpg')
            if args.net == 'mobileface':
                img = align(item, landmarks, (112, 112))
                #img = cv2.resize(img,(112,112))
                img = test_transform(img).reshape(1,3,112,112)
                if args.cuda:
                    img = img.cuda()
            else:
                img = np.asarray(align(item, landmarks, (96, 112)))
                img = img.transpose(2,0,1).reshape(1,3,112,96)
                img = (img-127.5)/128
                img = torch.from_numpy(img).float()
                if args.cuda:
                    img = img.cuda()
            final_imgs[id] = img
    if args.net == 'sphere':
        net = getattr(net_sphere,'sphere20a')()
        net.load_state_dict(torch.load(model_dir+'sphere20a_20171020.pth'))
    else:
        net = model.MobileFaceNet(512)
        net.load_state_dict(torch.load(model_dir + 'model_mobilefacenet.pth'))

    if args.cuda:
        net.cuda()
    net.eval()
    net.feature = True

    if args.phase == 'extra':
        for id,img in final_imgs.items():
            current_f = extract_feature(net,img)
            feature[id] = current_f
        f = open('feature'+'_'+args.net+'.pkl','wb')
        pickle.dump(feature,f)
        f.close()
        print ('done for extract feature,save as'+ ' feature'+'_'+args.net+'.pkl')
        sys.exit(0)
    else:
        count = 0
        wrong_list = []
        for test_id,img in final_imgs.items():
            fea = extract_feature(net,img)

            f = open('feature'+'_'+args.net+'.pkl','rb')
            gallery_feature = pickle.load(f)
            f.close()

            cur_max_p = -1
            cur_id = ''
            for id,cur_f in gallery_feature.items():
                current_cosdistance = cur_f[0].dot(fea[0])/(cur_f[0].norm()*fea[0].norm()+1e-5)
                if current_cosdistance > args.threshold:
                    print ('current pic{} may be include person{},cosdistance is {}'.format(test_id,id,current_cosdistance))
                if current_cosdistance >cur_max_p:
                    cur_max_p = current_cosdistance
                    cur_id = id
            print('current pic {} include {}, is \t '.format(test_id,cur_id) + str(cur_id.split('.')[0] == test_id.split('_')[0]) )

            if cur_id.split('.')[0] == test_id.split('_')[0]:
                count += 1
            else:
                wrong_list.append(str(test_id) + '\tas to\t' + str(cur_id))
            print ('-------------------------------')

        print ('***********\n','all samples: {}, correct samples: {}\n'.format(len(final_imgs),count),'accuracy : {}'.format(count/len(final_imgs)))
        print ('\nwrong sample:\n')
        for i in wrong_list:
            print (i)


