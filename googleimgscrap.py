#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 00:22:34 2019

@author: kevin
"""


import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

headers={
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
        }
def scrap_google_img(teamname,keyword,limit):
    path = r"/Users/kevin/Desktop/photo/"
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    link_list = []
        
    x = (limit / 50) - 5
    int(x)
    
    print("Starting...")
    driver = webdriver.Chrome(executable_path="/usr/local/bin/chromedriver", chrome_options=chrome_options)
    driver.get("http:google.com/images")
    
    searchbar = driver.find_element_by_name('q')
    searchbar.clear()
    searchbar.send_keys(teamname+' '+keyword)
    
    button = driver.find_element_by_class_name('Tg7LZd')
    button.click()
    
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
        # Wait to load page 
        time.sleep(1)
    
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    j=0
    while j < x:
        time.sleep(1)
        try:
            load = driver.find_element_by_id('smb');
            load.click()
        except:
            print ("Element not found and test failed")
    
        j+=1
    
    print("Grabbing Links...")
    
    i=1
    for i in range(limit):
        elem = driver.find_elements_by_xpath('//*[@id="rg_s"]/div['+str(i+1)+']/a[1]/img')
        if (elem[0].get_attribute("data-src")==None):
            link_list.append(elem[0].get_attribute("src"))
        else:
            link_list.append(elem[0].get_attribute("data-src"))

       
    print("Downloading Images to Folder...")
    j = 0
    opener = urllib.request.build_opener()
    opener.addheaders = [('user-agent','Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36')]
    urllib.request.install_opener(opener)
    while j<limit:
        j += 1
        print("Downloading Image"+str(j))
        urllib.request.urlretrieve(link_list[j-1], path+teamname+' '+keyword+str(j)+".jpg")

        
    
    print("done...")
    driver.close()
    
    
    
    
    '''while j<limit:
        j += 1
        #time.sleep(random.randint(1,3))
        try:
            	print("Downloading Image"+str(i))
            	urllib.request.urlretrieve(photourl_list[i], path+teamname+keyword+str(j)+".jpg")
        except:
            try:
                print('skip this')
            except:
                continue
        i += 1
    
    print("done...")
    driver.close()'''
    
    
    '''x=0
    for imgurl in link_list:
        try:
            imgres = requests.get(imgurl)
            with open(path+teamname+keyword+str(x)+".jpg",'wb') as f:
                f.write(imgres.content)
                x +=1
                print("download image {}".format(x))
        except:
            try:
                print('skip this')
                #j -= 1
            except:
                continue
 
    
    print("done...")
    driver.close()'''
    
    
    '''x=0
    for imgurl in photourl_list:
        try:
            request = urllib.request.Request(imgurl,headers=headers)
            response = urllib.request.urlopen(request)
            with open(path+teamname+keyword+str(x)+".jpg",'wb') as f:
                f.write(response.read())
                x +=1
                print("download image {}".format(x)) 
        except:
            try:
                print('skip this')
                #j -= 1
            except:
                continue
            
    print("done...")
    driver.close()'''