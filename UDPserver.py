#!python3
##Receive perfect image frames from clients###

import socket
import struct
import time
import numpy as np
import pickle
import cv2

address = ("", 9999)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(address)
print('SERVER STARTED RUNNING')

#winname = "haoxin"
#cv2.namedWindow(winname)

data = b''
frameid = 0
headerLength = 12
pkgLength = 10240
timeout = 0.004 #unit:second
toCountThsd = 2 # timeout count threshold
delimiter = "||"
newImageFlag = 0 ## avoid the server does not receive the first packet of an image
timeoutCount = 0
PerfectFrames = 0
BrokenFrames = 0

while True:
	while True:
		try:
			s.settimeout(timeout) ## set packet receiving timeout
			imagePkg, addr = s.recvfrom(pkgLength)
			if imagePkg:
				headerRaw = imagePkg.split(delimiter)[0]
				header = struct.unpack('!i', headerRaw)[0]
				print "header= ", header
				if header == 0:	## first packet		
					imageSizeRaw = imagePkg.split(delimiter)[1]
					imageSize = struct.unpack('!i', imageSizeRaw)[0]
					packetSize = imageSize
					print "Image size: ", imageSize
					newImageFlag = 1
				elif header == 1:
					if newImageFlag == 1:
						packetSizeRaw = imagePkg.split(delimiter)[1]
						packetSize = struct.unpack('!i', packetSizeRaw)[0]
					elif newImageFlag == 0:
						print "lost the header packet"
						break
				imagedataRaw = imagePkg.split(delimiter)[2] ##image data
				data += imagedataRaw #String
				print "data length: ", len(data)
				if (len(data) >= imageSize) or (packetSize < pkgLength-headerLength):
					if len(data) >= imageSize:
						PerfectFrames += 1
						print "complete frames number = ", PerfectFrames
					else:
						BrokenFrames += 1
						print "Broken frames number = ", BrokenFrames
					frame_data = data[:len(data)]
					data = data[len(data):]
					imgdata = np.fromstring(frame_data, dtype='uint8') #String to np.byte
					decimg = cv2.imdecode(imgdata,1)
					frameid += 1
					newImageFlag = 0
					timeoutCount = 0
					str1 = '0'
					s.sendto(str1, addr)
					cv2.imwrite("/home/nvidia/Desktop/haoxin/UDPtest/image%3d.jpg" %frameid,decimg)
					#cv2.imshow(winname,decimg)
					print "Case 1 Finish %d receiving %s bytes from %s" %(frameid,len(frame_data),addr)
					break
				#if cv2.waitKey(1) & 0xFF == ord('q'):
				#	break
		except socket.timeout:
			#print "timeout!"
			if (newImageFlag == 1) and (timeoutCount < toCountThsd):
				timeoutCount += 1 ##
				print "retry once!"
			elif (newImageFlag == 1) and (timeoutCount >= toCountThsd):
				BrokenFrames += 1
				frame_data = data[:len(data)]
				data = data[len(data):]
				imgdata = np.fromstring(frame_data, dtype='uint8')#String to np.byte
				decimg = cv2.imdecode(imgdata,1)
				frameid += 1
				newImageFlag = 0
				timeoutCount = 0
				str1 = '0'
				s.sendto(str1, addr)
				cv2.imwrite("/home/nvidia/Desktop/haoxin/UDPtest/image%3d.jpg" %frameid,decimg)
				print "Case2 Finish %d receiving %s bytes from %s" %(frameid,len(frame_data),addr)
				print "Broken frames number = ", BrokenFrames
				break

