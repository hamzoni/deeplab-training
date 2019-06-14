
import os
import random

import sys
print ("This is the name of the script: ", sys.argv[0])

xmlfilepath = 'image'   
txtsavepath = 'index'
total_xml = os.listdir(xmlfilepath)
indexPath = '/home/taquy/projects/test/deeplab/datasets/screw_seg/index'

num=len(total_xml)
list = range(num)


trainval = random.sample(list, num)  

os.chdir(indexPath)   

ftrainval = open('train.txt', 'w')  

for i in list :
  name =total_xml[i][:-4] + '\n'
  ftrainval.write(name)
ftrainval.close()