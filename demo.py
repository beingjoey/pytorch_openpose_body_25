#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:17:47 2020

@author: joe
"""


from src import torch_openpose,util

import cv2

if __name__ == "__main__":
    tp = torch_openpose.torch_openpose('body_25')
    img = cv2.imread("images/timg.jpeg")
    poses = tp(img)
    img = util.draw_bodypose(img, poses,'body_25')
    cv2.imshow('v',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()