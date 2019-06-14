import os
import random

import sys

DIR = sys.argv[1];

xmlfilepath = DIR + '/image'
txtsavepath = DIR + '/index'
total_xml = os.listdir(xmlfilepath)
indexPath = DIR + '/index'

num=len(total_xml)
list = range(num)

trainval = random.sample(list, num)  

os.chdir(indexPath)

ftrainval = open('train.txt', 'w')  

for i in list :
  name =total_xml[i][:-4] + '\n'
  ftrainval.write(name)
ftrainval.close()