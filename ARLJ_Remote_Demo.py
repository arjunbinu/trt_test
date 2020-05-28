import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import math
import random
import cv2
import keras
import threading
import numpy as np
import pytesseract
import traceback
import darknet.python.darknet as dn
import time
import timeit
import datetime
import csv

from multiprocessing import Queue,Process
from ctypes import *
from random import randint
from os.path import splitext, basename, isdir
from os import makedirs
from darknet.python.darknet import detect
from glob import glob
from src.label import dknet_label_conversion
from src.label import Label, lwrite
from src.utils import nms
from src.utils import crop_region, image_files_from_folder
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes


def write_train_file(data):
        tmpname = "./Snapshots/database.csv"
	csvfile =open(tmpname,"a",'utf-8')	
	filewriter =csv.writer(csvfile,delimiter='\t')
	filewriter.writerow(str(data))
	csvfile.close()

def show0(buff0):
	print("Show 0")
	cv2.namedWindow("cam1")
        cv2.moveWindow("cam1",0,400)

	while True:
		try:
			frame = buff0.get()
		except Queue.Empty:
			frame = None
		frame=cv2.resize(frame,(320,320))

		cv2.imshow("cam1",frame)
		if cv2.waitKey(1) == 27:
			os.system("./red_52_0.expect")
			os.system("./red_100_0.expect")
			os.system("./red_126_0.expect")
			break

def show1(buff1):
	print("Show 1")
	cv2.namedWindow("cam2")
        cv2.moveWindow("cam2",420,400)
	while True:
		try:
			frame = buff1.get()
		except Queue.Empty:
			frame = None
		frame=cv2.resize(frame,(320,320))

		cv2.imshow("cam2",frame)
		if cv2.waitKey(1) == 27:
			os.system("./red_52_0.expect")
			os.system("./red_100_0.expect")
			os.system("./red_126_0.expect")
			break


def show2(buff2):
	print("Show 2")
	cv2.namedWindow("cam3")
        cv2.moveWindow("cam3",850,400)
	while True:
		try:
			frame = buff2.get()
		except Queue.Empty:
			frame = None
		frame=cv2.resize(frame,(320,320))
		
		cv2.imshow("cam3",frame)
		if cv2.waitKey(1) == 27:
			os.system("./red_52_0.expect")
			os.system("./red_100_0.expect")
			os.system("./red_126_0.expect")
			break


def show3(buff3):
	print("Show 3")
	cv2.namedWindow("cam4")
        cv2.moveWindow("cam4",850,0)
	while True:
		try:
			frame = buff3.get()
		except Queue.Empty:
			frame = None
		frame=cv2.resize(frame,(320,320))
		cv2.imshow("cam4",frame)
		if cv2.waitKey(1) == 27:
			os.system("./red_52_0.expect")
			os.system("./red_100_0.expect")
			os.system("./red_126_0.expect")
			break

def show4(buff1):
	print("Show 3")
	cv2.namedWindow("cam4")
        cv2.moveWindow("cam4",1200,400)
	while True:
		try:
			frame = buff1.get()
		except Queue.Empty:
			frame = None
		frame=cv2.resize(frame,(320,320))
		cv2.imshow("cam4",frame)
		if cv2.waitKey(1) == 27:
			os.system("./red_52_0.expect")
			break

def read0(buff0,buff1):
		
	camlink0 = ('rtspsrc location=rtsp://192.168.1.52:8551/main latency=100 ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int)640, height=(int)480, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink')

	print("11111111111111111")
	camlink1 = ('rtspsrc location=rtsp://192.168.1.168:8551/main latency=100 ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int)640, height=(int)480, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink')

	print("22222222222222222")
	cap0 = cv2.VideoCapture(camlink0,cv2.CAP_GSTREAMER)
	cap1 = cv2.VideoCapture(camlink1,cv2.CAP_GSTREAMER)
	time.sleep(0.5)
	while True:
		_,frame0 = cap0.read()
		buff0.put(frame0)
		_,frame1 = cap1.read()
		buff1.put(frame1)

def lp_det(Detect_Queue, buff1,buff2,buff3):
	
	entry = True
	exit = False
	mydir = None
	#img_entry2 = None
	#img_entry3 = None
	def sample(probs):
	    s = sum(probs)
	    probs = [a/s for a in probs]
	    r = random.uniform(0, 1)
	    for i in range(len(probs)):
		r = r - probs[i]
		if r <= 0:
		    return i
	    return len(probs)-1

	def c_array(ctype, values):
	    arr = (ctype*len(values))()
	    arr[:] = values
	    return arr

	class BOX(Structure):
	    _fields_ = [("x", c_float),
		        ("y", c_float),
		        ("w", c_float),
		        ("h", c_float)]

	class DETECTION(Structure):
	    _fields_ = [("bbox", BOX),
		        ("classes", c_int),
		        ("prob", POINTER(c_float)),
		        ("mask", POINTER(c_float)),
		        ("objectness", c_float),
		        ("sort_class", c_int)]


	class IMAGE(Structure):
	    _fields_ = [("w", c_int),
		        ("h", c_int),
		        ("c", c_int),
		        ("data", POINTER(c_float))]

	class METADATA(Structure):
	    _fields_ = [("classes", c_int),
		        ("names", POINTER(c_char_p))]

	    


	class IplROI(Structure):
	    pass

	class IplTileInfo(Structure):
	    pass

	class IplImage(Structure):
	    pass

	IplImage._fields_ = [
	    ('nSize', c_int),
	    ('ID', c_int),
	    ('nChannels', c_int),               
	    ('alphaChannel', c_int),
	    ('depth', c_int),
	    ('colorModel', c_char * 4),
	    ('channelSeq', c_char * 4),
	    ('dataOrder', c_int),
	    ('origin', c_int),
	    ('align', c_int),
	    ('width', c_int),
	    ('height', c_int),
	    ('roi', POINTER(IplROI)),
	    ('maskROI', POINTER(IplImage)),
	    ('imageId', c_void_p),
	    ('tileInfo', POINTER(IplTileInfo)),
	    ('imageSize', c_int),          
	    ('imageData', c_char_p),
	    ('widthStep', c_int),
	    ('BorderMode', c_int * 4),
	    ('BorderConst', c_int * 4),
	    ('imageDataOrigin', c_char_p)]


	class iplimage_t(Structure):
	    _fields_ = [('ob_refcnt', c_ssize_t),
		        ('ob_type',  py_object),
		        ('a', POINTER(IplImage)),
		        ('data', py_object),
		        ('offset', c_size_t)]



	#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
	lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)
	lib.network_width.argtypes = [c_void_p]
	lib.network_width.restype = c_int
	lib.network_height.argtypes = [c_void_p]
	lib.network_height.restype = c_int

	predict = lib.network_predict
	predict.argtypes = [c_void_p, POINTER(c_float)]
	predict.restype = POINTER(c_float)

	set_gpu = lib.cuda_set_device
	set_gpu.argtypes = [c_int]

	make_image = lib.make_image
	make_image.argtypes = [c_int, c_int, c_int]
	make_image.restype = IMAGE

	get_network_boxes = lib.get_network_boxes
	get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
	get_network_boxes.restype = POINTER(DETECTION)

	make_network_boxes = lib.make_network_boxes
	make_network_boxes.argtypes = [c_void_p]
	make_network_boxes.restype = POINTER(DETECTION)

	free_detections = lib.free_detections
	free_detections.argtypes = [POINTER(DETECTION), c_int]

	free_ptrs = lib.free_ptrs
	free_ptrs.argtypes = [POINTER(c_void_p), c_int]

	network_predict = lib.network_predict
	network_predict.argtypes = [c_void_p, POINTER(c_float)]

	reset_rnn = lib.reset_rnn
	reset_rnn.argtypes = [c_void_p]

	load_net = lib.load_network
	load_net.argtypes = [c_char_p, c_char_p, c_int]
	load_net.restype = c_void_p

	do_nms_obj = lib.do_nms_obj
	do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

	do_nms_sort = lib.do_nms_sort
	do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

	free_image = lib.free_image
	free_image.argtypes = [IMAGE]

	letterbox_image = lib.letterbox_image
	letterbox_image.argtypes = [IMAGE, c_int, c_int]
	letterbox_image.restype = IMAGE

	load_meta = lib.get_metadata
	lib.get_metadata.argtypes = [c_char_p]
	lib.get_metadata.restype = METADATA

	load_image = lib.load_image_color
	load_image.argtypes = [c_char_p, c_int, c_int]
	load_image.restype = IMAGE

	rgbgr_image = lib.rgbgr_image
	rgbgr_image.argtypes = [IMAGE]

	predict_image = lib.network_predict_image
	predict_image.argtypes = [c_void_p, IMAGE]
	predict_image.restype = POINTER(c_float)

	def classify(net, meta, im):
	    out = predict_image(net, im)
	    res = []
	    for i in range(meta.classes):
		res.append((meta.names[i], out[i]))
	    res = sorted(res, key=lambda x: -x[1])
	    return res



	def array_to_image(arr):

	    arr = arr.transpose(2,0,1)
	    c, h, w = arr.shape[0:3]
	    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
	    data = arr.ctypes.data_as(POINTER(c_float))
	    im = IMAGE(w,h,c,data)
	    return im, arr

	def detect(net, meta, image, thresh=.7, hier_thresh=.5, nms=.45):
	    
	    im, image = array_to_image(image)
	    rgbgr_image(im)
	    num = c_int(0)
	    pnum = pointer(num)
	    predict_image(net, im)
	    dets = get_network_boxes(net, im.w, im.h, thresh, 
		                     hier_thresh, None, 0, pnum)
	    num = pnum[0]
	    if nms: do_nms_obj(dets, num, meta.classes, nms)

	    res = []
	    for j in range(num):
		a = dets[j].prob[0:meta.classes]
		if any(a):
		    ai = np.array(a).nonzero()[0]
		    for i in ai:
		        b = dets[j].bbox
		        res.append((meta.names[i], dets[j].prob[i], 
		                   (b.x, b.y, b.w, b.h)))

	    res = sorted(res, key=lambda x: -x[1])
	    wh = (im.w,im.h)
	    if isinstance(image, bytes): free_image(im)
	    free_detections(dets, num)
	    return res






	def runOnVideo(net, meta, vid_source, thresh=0.80, hier_thresh=0.4, nms=0.45):
	    video = cv2.VideoCapture(0)
	    count = 0

	    classes_box_colors = [(0, 0, 255), (0, 255, 0)] 
	    classes_font_colors = [(255, 255, 0), (0, 255, 255)]
	    while video.isOpened():        
		res, frame = video.read()
		if not res:
		    break        
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		im, arr = array_to_image(rgb_frame)
		
		num = c_int(0)
		pnum = pointer(num)
		predict_image(net, im)
		dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
		num = pnum[0]
		if (nms): do_nms_obj(dets, num, meta.classes, nms);
		# res = []
		for j in range(num):
		    for i in range(meta.classes):
		        if dets[j].prob[i] > 0:
		            b = dets[j].bbox
		            x1 = int(b.x - b.w / 2.)
		            y1 = int(b.y - b.h / 2.)
		            x2 = int(b.x + b.w / 2.)
		            y2 = int(b.y + b.h / 2.)
		            cv2.rectangle(frame, (x1, y1), (x2, y2), classes_box_colors[0], 2)
		            cv2.putText(frame, meta.names[i], (x1, y1 - 20), 1, 1, classes_font_colors[0], 2, cv2.LINE_AA)
		                    
		cv2.imshow('output', frame)
		if cv2.waitKey(1) == ord('q'):
		    break        
		# print res
		count += 1


	def adjust_pts(pts,lroi):
		return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

	

	def text_extract(img_path, height, width):

			

			#fp = open("./Snapshots/database.txt","w+")

				#print '\t\tLP: %s' % lp_str
			#else:
				#print 'No characters found'
			#y = cv2.imwrite(
			#img1 = cv2.imread(img_path)
			gray = cv2.cvtColor(img_path,cv2.COLOR_BGR2GRAY)
			#cv2.imwrite("gray.jpg",gray)
			
			gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

			inv = 255 - gray    
			horizontal_img = inv
			vertical_img = inv

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
			horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
			horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
			vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
			vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

			mask_img = horizontal_img + vertical_img
			no_border = np.bitwise_or(gray, mask_img)
			cv2.imwrite("no.jpg",no_border)
	#text = pytesseract.image_to_string(no_border, lang='jpn')

			os.system("convert "+" no.jpg" + " -bordercolor " + " White" + " -border" + " 10x10" +" nob.jpg")
			imagez = cv2.imread("nob.jpg")			
			s1 = (20,30)
			s2 = (60,15)
			
			(x1,y1) = (20,30)

			(x21,y21) = (60,15)
			

			e1 = (60,80) 
			e2 = (200,45)

			
			(x2,y2) = (60,80)
			(x22,y22) = (200,45)

			
	 
			color = (255, 255, 255) 

	  
			thickness = 2
	  

			image1 = cv2.rectangle(imagez, s1, e1, color, thickness)
			image2 = cv2.rectangle(imagez, s2, e2, color, thickness)
			


			roi1 = image1[y1:y2,x1:x2]
			#cv2.imwrite("1.jpg",roi1)
			text = pytesseract.image_to_string(roi1, config='-l jpn --oem 0 --psm 5')
			#print text
			roi2 = image2[y21:y22,x21:x22]
			cv2.imwrite("2.jpg",roi2)
			text2 = pytesseract.image_to_string(roi2, config='-l jpn --oem 0 --psm 6')
			#print text2
			t = text2[:2]
			print t
		        cdr1=os.getcwd()
			database = '/home/nvidia/Desktop/ARLS_TLRS_Backup/ARLJ_TLRS_ENTRY_EXIT/lp_database'
			os.chdir(database)
		        with open('%s.txt' % ('lp'),'a+') as f:
					f.write(text+text2+ '\n')
			os.chdir(cdr1)
			ocr_threshold = .4
			R = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)
			if len(R):
				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)
				L.sort(key=lambda x: x.tl()[0])
				lp_str1 = ''.join([chr(l.cl()) for l in L])
			ocr_threshold = .4
			img_path1=cv2.imread('2.jpg')
			R = detect(ocr_net, ocr_meta, img_path1 ,thresh=ocr_threshold, nms=None)
			if len(R):
				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)
				L.sort(key=lambda x: x.tl()[0])
				lp_str2 = ''.join([chr(l.cl()) for l in L])	
				an = lp_str2[-3:]
				an = str(an)
				if 'Z' in an:
					res = an.replace("Z","2")
                                        print(text+t+' '+res+' '+lp_str1)
					line=text+t+res+lp_str1
				else:
					line=text+t+an+lp_str1
					#line=lp_str1
                                        print(text+t+' '+an+' '+lp_str1)
				#write_train_file(line)
				

				#fp.write(line)
		

	def lp_detector(Ivehicle, wpod_net):

		lp_threshold = .5
		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)),608)
		#print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)
		Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
		if len(LlpImgs):
			Ilp = LlpImgs[0]
			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
			cv2.imwrite("temp.png",Ilp*255.)
		return len(LlpImgs)
		


	def car_detect(Iorig):

		Lcars=[]	
		R = detect(vehicle_net, vehicle_meta, Iorig)
		R = [r for r in R if r[0] in ['car','bus']]
		print '\t\t%d cars found' % len(R)
		if len(R):
			WH = np.array(Iorig.shape[1::-1],dtype=float)
			Lcars = []
			for i,r in enumerate(R):
				cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
				tl = np.array([cx - w/2., cy - h/2.])
				br = np.array([cx + w/2., cy + h/2.])
				label = Label(0,tl,br)
				Lcars.append(label)
		return Lcars





	def crop1(I,label,bg=0.5):

		wh = np.array(I.shape[1::-1])

		ch = I.shape[2] if len(I.shape) == 3 else 1
		#print label
		tl = np.floor(label.tl()*wh).astype(int)
		br = np.ceil (label.br()*wh).astype(int)
		outwh = br-tl
		if tl[0]<1:
			tl[0]=1
		elif tl[1]<1:
			tl[1]=1
		elif br[0]>dim1:
			br[0]=dim1-1
		elif br[1]>dim1:
			br[1]=dim2-1

		return tl, br, outwh





	ocr_weights = 'data/ocr/ocr-net.weights'
	ocr_netcfg  = 'data/ocr/ocr-net.cfg'
	ocr_dataset = 'data/ocr/ocr-net.data'
	ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
	ocr_meta = dn.load_meta(ocr_dataset)
	wpod_net = load_model("data/lp-detector/wpod-net_update1.h5")
	vehicle_threshold = .5
	vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
	vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
	vehicle_dataset = 'data/vehicle-detector/voc.data'
	vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
	vehicle_meta = dn.load_meta(vehicle_dataset)
	countd = 1
	while True:
		try:
			img = Detect_Queue.get()
			img_entry2 = buff2.get()
			img_entry3 = buff3.get()
		except :
			img = None
			img_entry1 = None
			img_entry2 = None
		dim1 = img.shape[0]
		dim2 = img.shape[1]

		if (countd % 15) == 0:
	    		if exit == True:
				start = timeit.default_timer()
		    		lab=car_detect(img)
				if(len(lab)!=0):
					for i in range(len(lab)):
					    top, bot, wh = crop1(img, lab[i])
					    #print top
					    print("exit")
					    print  bot
					    roi = img[top[1]:bot[1], top[0]:bot[0]]
					    lp1 = lp_detector(roi, wpod_net)

					    print("*************************************************************")
					    
					    if(bot[1] > 360 and bot[1] < 390 ):
						lp=cv2.imread("temp.png")
						height = lp.shape[0]
						width = lp.shape[1]
						text=text_extract(lp, height, width)
					    if(bot[1] > 390 and bot[1] < 410):
		                                top_view = buff1.get()
		                                date_string =time.strftime("%Y-%m-%d-%H:%M")
		       				cdr=os.getcwd()
						mydir = os.path.join(os.getcwd(),datetime.datetime.now().strftime('%Y-%m-%d'))
						isdr=os.path.isdir(mydir)
						if isdr == False:
							os.makedirs(mydir)
						os.chdir(mydir)
		                		date_string =time.strftime("%H:%M:%S")
						#cv2.imwrite((('./Snapshots/Vehicle' + date_string +'.png'),top_view)	
						cv2.imwrite(mydir + date_string +'top.png',top_view)	
						os.chdir(cdr)                                        
						os.system("./blue_52_1.expect")
					    if(bot[1] > 410 ):
						os.system("./red_52_1.expect")
						entry = True
			if entry == True:
		    		lab2=car_detect(img_entry2)
		    		lab3=car_detect(img_entry3)
				if(len(lab2)!=0):
					for i in range(len(lab2)):
					    top2, bot2, wh2 = crop1(img_entry2, lab2[i])

					    print("entry1") 
					    print bot2
					    roi2 = img_entry2[top2[1]:bot2[1], top2[0]:bot2[0]]
					    lp12 = lp_detector(roi2, wpod_net)

					    print("*************************************************************")
					    print("entry2")
					    if(bot2[1] > 360 and bot2[1] < 390 ):
						lp2=cv2.imread("temp.png")
						height2 = lp2.shape[0]
						width2 = lp2.shape[1]
						text2=text_extract(lp2, height2, width2)
					    if(bot2[1] > 390 and bot2[1] < 410): 
						print 'blue'                                   
						os.system("./blue_100_1.expect")
						print 'on'
					    if(bot2[1] > 410 ):
						os.system("./red_100_1.expect")
						exit = True
				if(len(lab3)!=0):
					for i in range(len(lab3)):
					    top3, bot3, wh3 = crop1(img_entry3, lab3[i])

					    print("entry2") 
					    print bot3
					    roi3 = img_entry3[top3[1]:bot3[1], top3[0]:bot3[0]]
					    lp13 = lp_detector(roi3, wpod_net)

					    print("*************************************************************")
					    print("entry3")
					    if(bot3[1] > 315 and bot3[1] < 330 ):
						lp3=cv2.imread("temp.png")
						height3 = lp3.shape[0]
						width3 = lp3.shape[1]
						text3=text_extract(lp3, height3, width3)
					    if(bot3[1] > 331 and bot3[1] < 380): 
						print 'blue'                                   
						os.system("./blue_126_1.expect")
						print 'on'
					    if(bot3[1] > 380 ):
						os.system("./red_126_1.expect")
						exit = True
										

		countd=countd +1            



		
def read1(buff1):
	camlink2 = ('rtspsrc location=rtsp://192.168.1.168:8551/main latency=100 ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int)640, height=(int)480, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink')


	cap = cv2.VideoCapture(camlink2,cv2.CAP_GSTREAMER)
	time.sleep(0.5)	
	while True:
		_,frame = cap.read()
		buff1.put(frame)
		
def read2(buff2,buff3):
	camlink2 = ('rtspsrc location=rtsp://192.168.1.100:8551/main latency=100 ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int)640, height=(int)480, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink')


	camlink3 = ('rtspsrc location=rtsp://192.168.1.126:8551/main latency=100 ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int)640, height=(int)480, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink')


	cap2 = cv2.VideoCapture(camlink2,cv2.CAP_GSTREAMER)
	cap3 = cv2.VideoCapture(camlink3,cv2.CAP_GSTREAMER)
	time.sleep(0.5)
	while True:
		_,frame2 = cap2.read()
		buff2.put(frame2)
		_,frame3 = cap3.read()
		buff3.put(frame3)
		
def read3(buff3):
	camlink4 = ('rtspsrc location=rtsp://192.168.1.126:8551/main latency=100 ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int)640, height=(int)480, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink')


	cap = cv2.VideoCapture(camlink4,cv2.CAP_GSTREAMER)
	time.sleep(0.5)
	while True:
		_,frame = cap.read()
		buff3.put(frame)
		


def main():
        snap = 0

        buff0 = Queue(100)
        buff1 = Queue(100)
        buff2 = Queue(100)
        buff3 = Queue(100)
	os.system("./red_52_1.expect")
	os.system("./red_100_1.expect")
	os.system("./red_126_1.expect")
        P0 = Process(target = read0,args = (buff0,buff1))
        P1 = Process(target = lp_det,args = (buff0,buff1,buff2,buff3))
        P2 = Process(target = read2,args = (buff2,buff3))	
        #P3 = Process(target = read3,args = (buff3,))
	

        P00 = Process(target = show0,args =(buff0,))
        P11 = Process(target = show1,args =(buff1,))
        P22 = Process(target = show2,args =(buff2,))
        P33 = Process(target = show3,args =(buff3,))


        P0.start()
        P2.start()
        #P3.start()
        time.sleep(2)
	P1.start()
        
	P00.start()
        P11.start()	
        P22.start()
        P33.start()


if __name__ == '__main__':
	main()

