#!/usr/bin/python

import socket
import struct
import time
import numpy as np
import pickle
import cv2

address = ("", 8888)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)
s.listen(5)

print('SERVER STARTED RUNNING')

client, addr = s.accept()
print 'got connected from', addr

data = b''
payload_size = struct.calcsize("L") 
frameid = 1
#winname = "haoxin"
#cv2.namedWindow(winname)

while True:
	#starttime_recv = time.time()
	buf = ''
	while len(buf)<4:
		buf += client.recv(4-len(buf))
	size = struct.unpack('!i', buf)[0]
	#print type(size)
	print "receiving %d bytes" % size	
	while len(data) < size:
		data += client.recv(4096)
	frame_data = data[:size]
	data = data[size:]
	imgdata = np.fromstring(frame_data, dtype='uint8')
	decimg = cv2.imdecode(imgdata,1)
	cv2.imwrite("/home/nvidia/Desktop/haoxin/TCPtest/image%3d.jpg" %frameid,decimg)
	print "Finish receiving %s bytes from %s" %(size,addr)
	
	str1 = '0'+'\n'
	client.sendall(str1.encode())
	frameid += 1

