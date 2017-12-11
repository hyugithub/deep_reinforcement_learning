#download all NIPS 2017 papers

import urllib
import re
#from io import StringIO
#import StringIO
from bs4 import BeautifulSoup
import urllib.request
import requests

url = "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-30-2017"
fp = urllib.request.urlopen(url)
mybytes = fp.read()
mystr = mybytes.decode("utf8")
fp.close()

#print(mystr)
#buf = StringIO.StringIO(myfile)
soup = BeautifulSoup(mystr, 'html.parser')

download_list = []   
prefix = "https://papers.nips.cc"
suffix = ".pdf"
for rows in soup.find_all('a', href=True):
    #print("Found the URL:", rows['href'])
    string = rows['href']
    if "paper" in string:
#        print(string)
        download_list.append(prefix+string+suffix)
        
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
    
