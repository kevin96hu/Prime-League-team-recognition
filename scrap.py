#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:37:48 2019

@author: kevin
"""

import requests
from bs4 import BeautifulSoup as bs
import re


headers={
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
        }

url_list = "https://soccer.hupu.com/england"
page_text1 = requests.get(url_list).text
soup1 = bs(page_text1,'html5lib')
team = soup1.find_all(href=re.compile('teams'),target='_blank')
a=r'https://soccer.hupu.com/teams/[0-9]{3,5}'
teamurl=re.findall(a,str(team[1:20]))

eachteamurl=dict()
for i in range(0,19): 
    page_text1 = requests.get(teamurl[i]).text
    soup1 = bs(page_text1,'html5lib')
    team = soup1.find_all('span')
    b=r'(?<=>).+?(?=<)'
    name=re.findall(b,str(team[0]))
    name=str(name[0]).split(' ',1)[1]
    team = soup1.find_all(href=re.compile('www'),target='_blank')
    a=r'(?<=>).*(?=<)'
    eachteamurl[name]=re.findall(a,str(team[5]))
    
eachteamurl['Brighton and Hove Albion'][0]='https://www.brightonandhovealbion.com/'
eachteamurl['Burnley'][0]='https://www.burnleyfootballclub.com/'
eachteamurl['Cardiff City'][0]='https://www.cardiffcityfc.co.uk/'
eachteamurl['Fulham'][0]='http://www.fulhamfc.com/'
eachteamurl['Leicester City'][0]='https://www.lcfc.com/'
eachteamurl['Liverpool'][0]='https://www.liverpoolfc.com/'
eachteamurl['Manchester City'][0]='https://www.mancity.com/'
eachteamurl['Wolverhampton Wanderers'][0]='https://www.wolves.co.uk/'
eachteamurl['Chelsea'][0]='https://www.chelseafc.com/en'
eachteamurl['Southampton'][0]='https://southamptonfc.com/'
for name in eachteamurl.keys():
    if name in ['Southampton','Wolverhampton Wanderers','Liverpool']:
        if name=='Southampton':
            eachteamurl[name][0]=eachteamurl[name][0]+'first-team'
        else:
            eachteamurl[name][0]=eachteamurl[name][0]+'team/first-team'
    else:
        if name=='Fulham':
            eachteamurl[name][0]=eachteamurl[name][0]+'first-team/player-profiles'
        else:
            if name=='Arsenal':
                eachteamurl[name][0]=eachteamurl[name][0]+'first-team/players'
            else: 
                if name=='Manchester United':
                    eachteamurl[name][0]=eachteamurl[name][0]+'/en/players-and-staff/first-team'
                else: 
                    if eachteamurl[name][0].endswith('/') :
                        eachteamurl[name][0]=eachteamurl[name][0]+'teams/first-team'
                    else:     
                        eachteamurl[name][0]=eachteamurl[name][0]+'/teams/first-team'

url_list = eachteamurl['Manchester City'][0]
page_text1 = requests.get(url_list).text
soup1 = bs(page_text1,'html5lib')
playername = soup1.find_all(class_='squad-listing--person-name')[0:26]
photoinfo = soup1.find_all(class_='lazyimage lazyload squad-listing--person-photo')[0:52]
c=r'(?<=>).+?(?=<)'
d=r'data-src=\"[a-zA-Z].+?\"'
Manchester_City=dict()
for i in range(0,len(playername)):   
    eachplayername=re.findall(c,str(playername[i]))
    eachplayerphoto=re.findall(d,str(photoinfo[2*i]))
    eachplayerphoto=eachplayerphoto[0].split('"')[1]
    Manchester_City[eachplayername[0]]=eachplayerphoto

url_list = eachteamurl['Liverpool'][0]
page_text1 = requests.get(url_list).text
soup1 = bs(page_text1,'html5lib')
photoinfo = soup1.find_all(class_='img-wrap')[0:28]
c=r'alt=\"[a-zA-Z].+?\"'
d=r'src=\"[a-zA-Z].+?\"'
Liverpool=dict()
for i in range(0,len(photoinfo)):   
    eachplayername=re.findall(c,str(photoinfo[i]))
    eachplayerphoto=re.findall(d,str(photoinfo[i]))
    eachplayername=eachplayername[0].split('"')[1]
    eachplayerphoto=eachplayerphoto[0].split('"')[1]
    Liverpool[eachplayername]=eachplayerphoto
    
url_list = eachteamurl['Tottenham Hotspur'][0]
page_text1 = requests.get(url_list).text
soup1 = bs(page_text1,'html5lib')
Tottenham_Hotspur = soup1.find_all(class_='Players__group')
playername = Tottenham_Hotspur[0].find_all(class_='PlayersPlayer__name')[0:27]
photoinfo = Tottenham_Hotspur[0].find_all(class_='PlayersPlayer__photo')[0:27]
c=r'(?<=>).+?(?=<)'
d=r'data-src=\"[a-zA-Z].+?(?=\*)'
Tottenham_Hotspur=dict()
for i in range(0,len(photoinfo)):   
    eachplayername=re.findall(c,str(playername[i]))
    eachplayerphoto=re.findall(d,str(photoinfo[i]))
    eachplayerphoto=eachplayerphoto[0].split('"')[1]
    Tottenham_Hotspur[eachplayername[0]]=eachplayerphoto

url_list = eachteamurl['Arsenal'][0]
page_text1 = requests.get(url_list).text
soup1 = bs(page_text1,'html5lib')
photoinfo = soup1.find_all(class_='player-card__portrait')
playername = soup1.find_all(class_='player-card__info')
eachplayername=[]
firstname=[]
lastname=[]
Arsenal=dict()
for i in range(0,len(playername)):   
    firstname.append(re.findall(r'(?<=>)(.+?)(?=<)',str(playername[i].find_all(class_='player-card__info__position-or-first-name'))))
    lastname.append(re.findall(r'(?<=>)(.+?)(?=<)',str(playername[i].find_all(class_='player-card__info__name'))))
    if(firstname[i]==['\xa0']):
        eachplayername.append(lastname[i][0])
    else:
        eachplayername.append(firstname[i][0]+' '+lastname[i][0])
for i in range(29):
    Arsenal[eachplayername[i]] = 'https://www.arsenal.com'+re.findall(r'data-src=\"([a-zA-Z/].*?)\"',str(photoinfo[i]))[0]

  
url_list = eachteamurl['Chelsea'][0]
page_text1 = requests.get(url_list,headers=headers).text
soup1 = bs(page_text1,'html5lib')
photoinfo = soup1.find_all(class_='tile__inner')
playername = soup1.find_all(class_='tile__description__heading')
eachplayername=[]
for i in range(0,len(playername)):  
    if (len(re.findall(r'(?<=inserted">)(.+?)(?=</)',str(playername[i])))==1):
        eachplayername.append(re.findall(r'(?<=>)(.+?)(?=<)',str(playername[i]))[0])
    elif (i==24):
        eachplayername.append(re.findall(r'(?<=inserted">)(.+?)(?=</)',str(playername[i]))[0])
    else:
        eachplayername.append(re.findall(r'(?<=inserted">)(.+?)(?=</)',str(playername[i]))[0]+' '+re.findall(r'(?<=inserted">)(.+?)(?=</)',str(playername[i]))[1])






from googleimgscrap import scrap_google_img as sgi
for name in list(Liverpool.keys()):
    sgi('Liverpool',name,name+' face -pes -fifa',15)
    sgi('Liverpool',name,name,50)
for name in list(Manchester_City.keys()):
    sgi('Manchester_City',name,name+' face -pes -fifa',15)
    sgi('Manchester_City',name,name,50)
for name in list(Tottenham_Hotspur.keys()):
    sgi('Tottenham_Hotspur',name,name+' face -pes -fifa',15)
    sgi('Tottenham_Hotspur',name,name,50)
for name in list(Arsenal.keys()):
    sgi('Arsenal',name,name+' face -pes -fifa',15)
    sgi('Arsenal',name,name,50)
for name in eachplayername:
    sgi('Arsenal',name,name+' face -pes -fifa',15)
    sgi('Arsenal',name,name,50)






