#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 01:51:12 2019

@author: kevin
"""

from PIL import Image
import os

for player in os.listdir('/Users/kevin/Desktop/photo4_face'):
    os.makedirs('/Users/kevin/Desktop/photoface4/'+player)
    for pic in os.listdir('/Users/kevin/Desktop/photo4_face/'+player):
        try:
            img=Image.open('/Users/kevin/Desktop/photo4_face/'+player+'/'+pic)
            
            if img.mode == 'RGBA':
                r, g, b, a = img.split()
                img = Image.merge("RGB", (r, g, b))
    
            dst = img.resize((128, 128))
            dst.save('/Users/kevin/Desktop/photoface4/'+player+'/'+pic)
        except:
            pass






