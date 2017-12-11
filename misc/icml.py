# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:15:17 2017

@author: HYU
"""

import urllib
import re
#from io import StringIO
#import StringIO
from bs4 import BeautifulSoup
import urllib.request
import requests

url = "http://proceedings.mlr.press/v70/"
fp = urllib.request.urlopen(url)
mybytes = fp.read()
mystr = mybytes.decode("utf8")
fp.close()

#print(mystr)
#buf = StringIO.StringIO(myfile)
soup = BeautifulSoup(mystr, 'html.parser')

download_list = []   
for rows in soup.find_all('a', href=True):
    #print("Found the URL:", rows['href'])
    string = rows['href']
    if "http://proceedings.mlr.press/v70/" in string \
        and "pdf" in string:
        print(string)
        download_list.append(string)

for link in download_list:
    fname = link.split("/")    
    fname = fname[-1]
    fname = fname.replace("-","_")
    #print(fname)
    req = requests.get(link)
    file = open(fname, 'wb')
    for chunk in req.iter_content(500000):
        file.write(chunk)
    file.close()    
