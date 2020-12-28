from matplotlib.colors import hsv_to_rgb
from pytesseract import image_to_string
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import inspect
import imutils
import logging  
import sys

import cv2
import re
import os

def init(argv2):
    directory = './log'
    writeImg = './img/'
    videoPath = os.path.abspath(argv2)
    
    filename = videoPath.replace('/', '_').replace(".", "_").replace("\\",'_').replace(":",'_')
    sucDir = writeImg + filename + '/suc/'
    failDir = writeImg + filename + '/fail/'

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(sucDir):
        os.makedirs(sucDir)
    if not os.path.exists(failDir):
        os.makedirs(failDir)

    fo = open(directory+'/'+filename+".txt", "wb")

    return sucDir, failDir,videoPath,fo

def getPercentileBlack(image,algorithm):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    list = [int(val[0]) for idx, val in enumerate(hist)]

    _,_,v_split=cv2.split(hsv)
    arr_std = np.std(v_split,ddof=1)
    arr_mean = np.mean(v_split)
    percentile = int(np.percentile(v_split,30))

    upper = (list.index(max(list))/2 + percentile) / 2

    #print list.index(max(list))/2, percentile

    mask = cv2.inRange(hsv, (0,0,0), (0,0,upper))

    cv2.bitwise_not(mask, mask);
    maskd_img = cv2.bitwise_and(gray, gray, mask=mask)

    if algorithm == 0:
        ret_img = maskd_img
    elif algorithm == 1:
        ret_img = mask    
    

    '''
    plt.subplot(2, 1, 1)
    plt.imshow(gray, cmap="gray")  

    plt.subplot(2, 1, 2)
    plt.imshow(ret_img, cmap="gray")

    plt.draw()
    plt.pause(3)
    plt.close('all')
    
    '''
    return ret_img

def getContoursImg(oriImage,processedImage):
    ret,thresh = cv2.threshold(processedImage,127,255,0)
    _,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    width = 1
    height = 1
    widthA = 1
    heightA = 1

    weightsA = -9999
    weightsB = -9999
    retImageA = []
    retImageB = []

    # For each contour, find the bounding rectangle and draw it
    for component in contours:
        currentContour = component
        
        rect = cv2.minAreaRect(currentContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        sp = box.shape
        #print box
        #print sp[0], sp[1]
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        #hight = y2 - y1
        #width = x2 - x1
        #crop_img= img[y1:y1+hight, x1:x1+width]
        
        w = x2 - x1
        h = y2 - y1
        x = x1
        y = y1
        if w < 10 or h < 10:
            continue

        ratio =float(w) / float(h)

        if (ratio > 1.89 or ratio < 1.50 or w < oriImage.shape[1] / 30 or h < oriImage.shape[0] / 30 or w > oriImage.shape[1] - 10 or h > oriImage.shape[0] - 10 or x+w >= oriImage.shape[1] or y+h >= oriImage.shape[0]):
            continue
            
        #print (w, h)
        #cv2.drawContours(oriImage,[box],0,(0,0,255),2)
        #cv2.rectangle(oriImage,(x,y),(x+w,y+h),(255,0,),3)

        logging.debug('Line%d:%s:\tcurrent rect ratio %f' % \
            (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name, ratio))

        try:
            rgb = cv2.cvtColor(thresh[int(y):int(y+h), int(x):int(x+w)], cv2.COLOR_GRAY2RGB)
            rgbAbove = cv2.cvtColor(thresh[int(y):int(y+h/3), int(x):int(x+w)], cv2.COLOR_GRAY2RGB)
            rgbCenter = cv2.cvtColor(thresh[int(y+h/3):int(y+h*2/3), int(x):int(x+w)], cv2.COLOR_GRAY2RGB)
            rgbBelow = cv2.cvtColor(thresh[int(y+h*2/3):int(y+h), int(x):int(x+w)], cv2.COLOR_GRAY2RGB)

            rgbEdgeA = cv2.cvtColor(thresh[int(y):int(y+h), int(x):int(x+w/10)], cv2.COLOR_GRAY2RGB)
            rgbEdgeB = cv2.cvtColor(thresh[int(y):int(y+h), int(x+w-w/10):int(x+w)], cv2.COLOR_GRAY2RGB)
            rgbEdgeC = cv2.cvtColor(thresh[int(y):int(y+h/10), int(x):int(x+w)], cv2.COLOR_GRAY2RGB)
            rgbEdgeD = cv2.cvtColor(thresh[int(y+h-h/10):int(y+h), int(x):int(x+w)], cv2.COLOR_GRAY2RGB)
            '''
            plt.subplot(5, 1, 1)
            plt.imshow(thresh[int(y):int(y+h), int(x):int(x+w/10)], cmap="gray")
            plt.subplot(5, 1, 2)
            plt.imshow(thresh[int(y):int(y+h), int(x+w-w/10):int(x+w)], cmap="gray")
            plt.subplot(5, 1, 3)
            plt.imshow(thresh[int(y):int(y+h/10), int(x):int(x+w)], cmap="gray")
            plt.subplot(5, 1, 4)
            plt.imshow(thresh[int(y+h-h/10):int(y+h), int(x):int(x+w)], cmap="gray")
            plt.subplot(5, 1, 5)
            plt.imshow(thresh[int(y):int(y+h), int(x):int(x+w)], cmap="gray")
            
            plt.draw()
            plt.pause(3)
            plt.close('all')
            '''

            #rgb = cv2.cvtColor(processedImage[box], cv2.COLOR_GRAY2RGB)

            hsvAbove = cv2.cvtColor(rgbAbove, cv2.COLOR_RGB2HSV)
            hsvCenter = cv2.cvtColor(rgbCenter, cv2.COLOR_RGB2HSV)
            hsvBelow = cv2.cvtColor(rgbBelow, cv2.COLOR_RGB2HSV)

            hsvEdgeA = cv2.cvtColor(rgbEdgeA, cv2.COLOR_RGB2HSV)
            hsvEdgeB = cv2.cvtColor(rgbEdgeB, cv2.COLOR_RGB2HSV)
            hsvEdgeC = cv2.cvtColor(rgbEdgeC, cv2.COLOR_RGB2HSV)
            hsvEdgeD = cv2.cvtColor(rgbEdgeD, cv2.COLOR_RGB2HSV)

            hA_split,sA_split,vA_split=cv2.split(rgbAbove)
            hB_split,sB_split,vB_split=cv2.split(hsvBelow)
            hC_split,sC_split,vC_split=cv2.split(hsvCenter)
            
            hEA_split,sEA_split,vEA_split=cv2.split(hsvEdgeA)
            hEB_split,sEB_split,vEB_split=cv2.split(hsvEdgeB)
            hEC_split,sEC_split,vEC_split=cv2.split(hsvEdgeC)
            hED_split,sED_split,vED_split=cv2.split(hsvEdgeD)

        except:
            continue

        #cv2.rectangle(oriImage,(x1,y1),(x2,y2),(255,0,),3)

        '''print (hA_split,sA_split,vA_split)
        print (hB_split,sB_split,vB_split)
        print ('--------------------------')
        '''
        weights = 0
        weightsAbove = 0
        weightsBelow = 0
        weightsCenter = 0
        weightsEdgeX = 0
        weightsEdgeY = 0

        for v1 in vA_split:
            for v2 in v1:
                weightsAbove += 1 if v2 == 0 else (-1)
        
        for v1 in vB_split:
            for v2 in v1:
                weightsBelow += 1 if v2 == 0 else (-1)

        for v1 in vC_split:
            for v2 in v1:
                weightsCenter += 1 if v2 == 255 else (-1)

        for v1 in vEA_split:
            for v2 in v1:
                weightsEdgeX += 1 if v2 == 255 else 0
                weightsEdgeY += 1 if v2 == 0 else 0
        for v1 in vEB_split:
            for v2 in v1:
                weightsEdgeX += 1 if v2 == 255 else 0
                weightsEdgeY += 1 if v2 == 0 else 0
        for v1 in vEC_split:
            for v2 in v1:
                weightsEdgeX += 1 if v2 == 255 else 0
                weightsEdgeY += 1 if v2 == 0 else 0
        for v1 in vED_split:
            for v2 in v1:
                weightsEdgeX += 1 if v2 == 255 else 0
                weightsEdgeY += 1 if v2 == 0 else 0

        if (weightsAbove < 0 or weightsBelow < 0 or float(weightsEdgeX) / float(weightsEdgeY) > 0.95):
            continue
        
        #print (weightsEdgeX, weightsEdgeY,  float(weightsEdgeX)/float(weightsEdgeY),w, h)
        weights = weightsAbove + weightsBelow + weightsCenter
        
        #print (ratio)
        '''
        plt.subplot(1, 1, 1)
        plt.imshow(processedImage[int(y):int(y+h), int(x):int(x+w)], cmap="gray")

        plt.draw()
        plt.pause(3)
        plt.close('all')
        '''
        
        #print weights , weightsAbove , weightsBelow
        logging.debug('Line%d:%s: this rect weights %f' % \
            (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name, weights))

        if weights > weightsA or weights > weightsB:
            if weightsA < weightsB:
                weightsA = weights
                #M = cv2.getRotationMatrix2D((h/2,w/2),rect[2],1)
                #res = cv2.warpAffine(oriImage[y:(y+h), (x):(x+w)],M,(w,h))

                #retImageA = res[int(0+h/3):int(0+h*2/3), int(0+5):int(0+w-5)] 

                retImageA = oriImage[int(y+h/3):int(y+h*2/3), int(x):int(x+w)]
                #retImageA = oriImage[y:(y+h), (x):(x+w)]
                #print 'A' 
                #print float(w) / float(h)
                #retImageA = oriImage[y1:y1+hight, x1:x1+width]
            else:
                weightsB = weights
                retImageB = oriImage[int(y+h/3):int(y+h*2/3), int(x):int(x+w)]
                #retImageB = oriImage[y:(y+h), (x):(x+w)]
                #print 'B' 
                #print float(w) / float(h)
                #retImageB = oriImage[y1:y1+hight, x1:x1+width]


        #cv2.drawContours(oriImage, currentContour, -1, (255, 0, 0), 4)
        cv2.rectangle(oriImage,(x,y),(x+w,y+h),(0,0,0),3)

    '''    
    plt.subplot(1, 1, 1)
    plt.imshow(oriImage, cmap="gray")
    plt.subplot(3, 1, 2)
    plt.imshow(retImageA, cmap="gray")
    plt.subplot(3, 1, 3)
    plt.imshow(retImageB, cmap="gray")

    plt.draw()
    plt.pause(3)
    plt.close('all')
    '''
    

    if weightsA > weightsB:
        return retImageA, retImageB
    else:
        return retImageB, retImageA

def getDelayTime(image, count, sucDir, failDir, algorithm):
    maskedImage = getPercentileBlack(image,algorithm)

    contoursA, contoursB = getContoursImg(image,maskedImage)

    result = -1
    stringA = '~'
    stringB = '~'
    if (type(contoursA).__module__ == 'numpy' and type(contoursB).__module__ == 'numpy'):
        stringA = image_to_string(contoursA, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789:')
        stringB = image_to_string(contoursB, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789:')
        stringA = stringA.replace(",", ".").replace("-", ".").replace(";", ":").replace("\'", ":").replace("!", ":").replace(" ", "");
        stringB = stringB.replace(",", ".").replace("-", ".").replace(";", ":").replace("\'", ":").replace("!", ":").replace(" ", "");


        if (stringA.find(':') != -1):
            stringA = '00' + stringA[stringA.find(':'):]
        else:
            stringA = '~'

        if (stringB.find(':') != -1):
            stringB = '00' + stringB[stringB.find(':'):]
        else:
            stringB = '~'

        if (len(stringA)>=12):
            stringA = stringA[0:12]
            stringA = stringA[:8] + '.' + stringA[9:]
        else:
            stringA = '~'

        if (len(stringB)>=12):
            stringB = stringB[0:12]
            stringB = stringB[:8] + '.' + stringB[9:]
        else:
            stringB = '~'

        if (re.match(r'\d+:\d+:\d+\.+\d', stringA) and re.match(r'\d+:\d+:\d+\.+\d', stringB)\
           and len(stringA) == 12 and len(stringB) == 12):
            bNumA = re.search('[a-zA-Z]', stringA)
            bNumB = re.search('[a-zA-Z]', stringB)
            bSymbolA = re.search('[%]', stringA)
            bSymbolB = re.search('[%]', stringB)
            if (re.search('[a-zA-Z]', stringA) == None and re.search('[a-zA-Z]', stringB) == None\
             and re.search('[%]', stringA) == None and re.search('[%]', stringB) == None):
                try:
                    datetimeA = datetime.strptime(stringA, '%H:%M:%S.%f')
                    datetimeB = datetime.strptime(stringB, '%H:%M:%S.%f')
                    micLenA = len(str(datetimeA.microsecond))
                    micLenB = len(str(datetimeB.microsecond))
                    result = abs(datetimeB - datetimeA)
                    cv2.imwrite(sucDir + str(count) + 'A' + ".jpg", contoursA)
                    cv2.imwrite(sucDir + str(count) + 'B' + ".jpg", contoursB)
                except:
                    result = -1
            else:
                result = -1
        else:
            result = -1
            cv2.imwrite(failDir + str(count) + 'A' + ".jpg", contoursA)
            cv2.imwrite(failDir + str(count) + 'B' + ".jpg", contoursB)

    if result != -1:
        logging.debug('Line%d:%s:\t[the %d frame success : %.3fs, %s, %s] \t%s' % \
            (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name,\
            count, result.total_seconds(), stringA, stringB, str(datetime.now())))
    else:
        logging.debug('Line%d:%s:\t[the %d frame failure : %s, %s, %s] \t%s' % \
            (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name,\
            count, str(-1), stringA, stringB, str(datetime.now())))

    return result, stringA, stringB

def main():
    #set log level [DEBUG INFO WARNING ERROR CRITICAL]
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    #start
    logging.info('Line%d:%s: \t[python %s %s] \t%s' % \
        (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name, sys.argv[0], sys.argv[1], str(datetime.now())))

    #create img/log dir
    sucDir, failDir, videoPath, fo = init(sys.argv[1])

    algorithm = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    
    #get video image
    videoCap = cv2.VideoCapture(videoPath)
    success,image = videoCap.read()
    count = 1
    
    #success = True
    #image = cv2.imread('IMG_20181229_214013.jpg')
    
    while success:
        logging.debug('Line%d:%s:\t[the %d frame load success] \t%s' % \
            (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name,count,str(datetime.now())))
        stringA = '~'
        stringB = '~'
        if (type(image).__module__ == 'numpy'):# and count > 6
            result, stringA, stringB = getDelayTime(image, count, sucDir, failDir,algorithm)

            if result != -1:
                logging.info('Line%d:%s:\t[the %d frame success : %.3fs, %s, %s] \t%s' % \
                    (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name,\
                    count, result.total_seconds(), stringA, stringB, str(datetime.now())))
                strWrite = str(count)+ ' ' + str(result.total_seconds()) + '\t' + stringA + '\t' + stringB + '\n'            
                fo.write( strWrite.encode("utf-8"));
            else:
                logging.info('Line%d:%s:\t[the %d frame failure : %s, %s] \t%s' % \
                    (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name,count, stringA, stringB,str(datetime.now())))
                strWrite = str(count)+ ' ' + str(-1) + '\t' + stringA + '\t' + stringB + '\n'  
                fo.write( strWrite.encode("utf-8"));
        else:
            logging.info('Line%d:%s:\t[the %d frame failure : %s, %s] \t%s' % \
                (inspect.currentframe().f_lineno, inspect.currentframe().f_code.co_name,count, stringA, stringB,str(datetime.now())))
            strWrite = str(count)+ ' ' + str(-1) + '\t' + stringA + '\t' + stringB + '\n'  
            fo.write( strWrite.encode("utf-8"));
        
        success,image = videoCap.read()
        count += 1

        #if (count > 9):
        #    break

    fo.close()


if __name__ == '__main__':
    main()
    
