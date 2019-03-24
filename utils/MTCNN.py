import os
os.environ['GLOG_minloglevel'] = '2' 

import numpy as np
import cv2
import caffe
import sys
import re
from scipy.misc import imresize
import copy
from model import predict, image_to_tensor, deepnn
import tensorflow as tf

INFERENCE = True
EXPRESSION_DETECTION_ENABLED = False 
    
minsize = 20
output_image = False
output_print = False 


caffe_model_path = "models/detection"

threshold = [0.9, 0.9, 0.95]
factor = 0.709

caffe.set_mode_gpu()
PNet = caffe.Net(caffe_model_path + "/det1.prototxt", caffe_model_path + "/det1.caffemodel", caffe.TEST)
RNet = caffe.Net(caffe_model_path + "/det2.prototxt", caffe_model_path + "/det2.caffemodel", caffe.TEST)
ONet = caffe.Net(caffe_model_path + "/det3.prototxt", caffe_model_path + "/det3.caffemodel", caffe.TEST)

owd = os.getcwd()
face_model_path = 'models/recognition/'

face_model = './face_deploy.prototxt'
face_weights = './85_accuracy.caffemodel'
center_facenet = caffe.Net(face_model_path + face_model, face_model_path + face_weights, caffe.TEST)

if EXPRESSION_DETECTION_ENABLED:
    face_x = tf.placeholder(tf.float32, [None, 2304])
    y_conv = deepnn(face_x)
    probs = tf.nn.softmax(y_conv)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('models/expression/ckpt')
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')
        
        

def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im


def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print("bb", boundingbox)
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print('#################')
    #print('boxes', boxes)
    #print('w,h', w, h)
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    #print('tmph', tmph)
    #print('tmpw', tmpw)

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)
    
   
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    
    #print('bboxA', bboxA)
    #print('w', w)
    #print('h', h)
    #print('l', l)
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x
    
    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet
        
        #print("1: x,y", x,y)
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print("2: x,y", x,y)
    else:
        score = map[x,y]
    '''
    #print("dx1.shape", dx1.shape)
    #print('map.shape', map.shape)
   

    score = map[x,y]
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    #print('(x,y)',x,y)
    #print('score', score)
    #print('reg', reg)

    return boundingbox_out.T






def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    
    img2 = img.copy()

    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m
          
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    # first stage
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]
        #im_data = imResample(img, hs, ws); print("scale:", scale)


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype = np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
    
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         



    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        #print("[2]:",total_boxes.shape[0])
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T
        #print("[3]:",total_boxes.shape[0])
        #print(regh)
        #print(regw)
        #print('t1',t1)
        #print(total_boxes)

        total_boxes = rerec(total_boxes) # convert box to square
        #print("[4]:",total_boxes.shape[0])
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        #print("[4.5]:",total_boxes.shape[0])
        #print(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    #print(total_boxes.shape)
    #print(total_boxes)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        #print('tmph', tmph)
        #print('tmpw', tmpw)
        #print("y,ey,x,ex", y, ey, x, ex, )
        #print("edy", edy)

        #tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            # New code sachin
            #tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
            if (tmph[k] > 2*img.shape[0] or tmpw[k] > 2*img.shape[1]):
                tempimg[:,:,:,k] = np.zeros(24,24,3);
            else:
                tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))


          
            #print("dx[k], edx[k]:", dx[k], edx[k])
            #print("dy[k], edy[k]:", dy[k], edy[k])
            #print("img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape)
            #print("tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape)

            #print("y,ey,x,ex", y[k], ey[k], x[k], ex[k])
            #print("tmp", tmp.shape)
            
            #tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            #print('tempimg', tempimg[k,:,:,:].shape)
            #print(tempimg[k,0:5,0:5,0] )
            #print(tempimg[k,0:5,0:5,1] )
            #print(tempimg[k,0:5,0:5,2] )
            #print(k)
    
        #print(tempimg.shape)
        #print(tempimg[0,0,0,:])
        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        #np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        #print(tempimg[0,:,0,0])
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        #print(out['conv5-2'].shape)
        #print(out['prob1'].shape)

        score = out['prob1'][:,1]
        #print('score', score)
        pass_t = np.where(score>threshold[1])[0]
        #print('pass_t', pass_t)
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        #print("[5]:",total_boxes.shape[0])
        #print(total_boxes)

        #print("1.5:",total_boxes.shape)
        
        mv = out['conv5-2'][pass_t, :].T
        #print("mv", mv)
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print('pick', pick)
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                #print("[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                #print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                #print("[8]:",total_boxes.shape[0])
            
        #####
        # 2 #
        #####
        #print("2:",total_boxes.shape)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
           
            #print('tmpw', tmpw)
            #print('tmph', tmph)
            #print('y ', y)
            #print('ey', ey)
            #print('x ', x)
            #print('ex', ex)
        

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                if (tmph[k] > 2*img.shape[0] or tmpw[k] > 2*img.shape[1]):
                    tempimg[:,:,:,k] = np.zeros(48,48,3)
                else:
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                    tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                    tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            #print("[9]:",total_boxes.shape[0])
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                #print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print(pick)
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                   #print("[11]:",total_boxes.shape[0])
                    points = points[pick, :]


        # LNet starts here. Sachin addition
    #    numbox = total_boxes.shape[0]
    #    if numbox > 0:
    #        tempimg = np.zeros((numbox, 24, 24, 15)) # (24, 24, 15, numbox)
    #        patchw = np.concatenate(np.array([total_boxes[:,3] - total_boxes[:,1] + 1 , total_boxes[:,2] - total_boxes[:,0]+1]).T)
    #        #print "sac", xaxa
    #        patchw = max(patchw)
    #        patchw = np.fix(patchw)
    #        tmp = np.where(np.mod(patchw, 2) == 1)
    #        # TODO
    #        patchw = patchw+1;
    #        #print "sac", numbox
    #        pointx = np.ones((numbox,5))
    #        pointy = np.ones((numbox,5))
    #        print total_boxes
    #        #print points
    #        for k in range(0, 5):
    #            tmp = np.matrix([points[:, k], points[:, k+5]])
    #            x = np.fix(tmp[0,:]-0.5*patchw)
    #            y = np.fix(tmp[1,:]-0.5*patchw)
    #            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)



    return total_boxes, points

count_num = 0
def process_image(ip_img):
    img=ip_img 
    img_bkup = img
    
    alignedFaces = []
    img_matlab = np.copy(img)
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp
    
    # Getting the bounding box and the landmarks and drawing the bounding box around the image
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    x1 = boundingboxes[:,0] 
    #print(x1)
    
    # Code for aligning the detected faces
    imgSize = (112, 96)
    # This part is needed by the DeepFace code
    x_ = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299]
    y_ = [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]
    src = np.array( zip(x_, y_) ).astype(np.float32).reshape(1,5,2)
    
    out = None
    emotion = ""
    features = []
    name_dict_arr = [] 
    for i in range(0, len(x1)):
        # Drawing the landmarks
        x = points[i][0:5]
        y = points[i][5:10]
    
        # Code for alginign the faces
        dst = np.array( zip(x, y) ).astype(np.float32).reshape(1,5,2)
        transmat = cv2.estimateRigidTransform( dst, src, False )

        if(transmat is not None):
            sys.stdout.flush()
            print "Person found"
            out = cv2.warpAffine(img_bkup, transmat, (imgSize[1], imgSize[0]))
            # DEBUG cv2.imwrite("new"+str(count_num)+".jpg", out)
            alignedFaces.append(out)
            if INFERENCE:
                npstore = out 
                npstore = npstore.astype(float)
                npstore = (npstore - 127.5)/128
                npstore = np.expand_dims(npstore, axis=0)
                npstore = np.swapaxes(npstore,1,3)
                npstore = np.swapaxes(npstore,2,3)
                
                center_facenet.blobs['data'].data[...] = npstore
                
                center_facenet.forward()
                features.append(copy.deepcopy(center_facenet.blobs["fc5"].data[0]))


                # Face expression part
                if EXPRESSION_DETECTION_ENABLED:
                    face_coor = np.ceil(boundingboxes[i]).astype(int)

                    image = img[face_coor[1]:face_coor[3], face_coor[0]:face_coor[2]]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #print("coord size", len(face_coor))

                    try:
                        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
                        import scipy.misc
                        scipy.misc.imsave('outfile.jpg', image)
                    except Exception:
                        continue
                    tensor = image_to_tensor(image)
                    result = sess.run(probs, feed_dict={face_x: tensor})
                    print("debug", result, np.argmax(result[0]))
                    emotion = EMOTIONS[np.argmax(result[0])]
        else:
            print "none detected"
    
    return alignedFaces, boundingboxes, features, emotion 

if __name__ == "__main__":
    # Train directory
    input_dir = "/home/sachin/Machine_Learning_Projects/datasets/CASIA-WebFace/"
    entity_list = os.listdir(input_dir)
    for person in entity_list:
        files = os.listdir(input_dir + person)
        for i, f in enumerate(files):
            return_array, _= process_image(cv2.imread(input_dir + person + "/" + f))
            if len(return_array) > 1:
                continue

            for j, out in enumerate(return_array):
                cv2.imwrite(input_dir + person + "/" + f[:-4] + "_aligned.jpg", out)
