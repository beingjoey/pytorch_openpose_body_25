#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:13:51 2020

@author: joe
"""
import numpy as np
import math

jointpairs = [[1,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[8,12],[12,13],[13,14],[0,15],[0,16]\
              ,[15,17],[16,18],[11,24],[11,22],[14,21],[14,19],[22,23],[19,20]]
#[[1,0],   [1,2],   [2,3],   [3,4],   [1,5],   [5,6],   [6,7],   [1,8], [8,9], [9,10],[10,11], [8,12],[12,13], [13,14], [0,15],  [0,16]]

#[[30, 31],[14, 15],[16, 17],[18, 19],[22, 23],[24, 25],[26, 27],[0, 1],[6, 7],[2, 3],[4, 5],  [8, 9],[10, 11],[12, 13],[32, 33],[34, 35]]

#[[15,17],[16,18],[11,24],[11,22],[14,21],[14,19],[22,23],[19,20]]
#[[36,37],[38,39],[50,51],[46,47],[44,45],[40,41],[48,49],[42,43]]
map25 = [[i,i+1] for i in range(0,52,2)]

def findoutmappair(all_peaks,paf):
    mid_num = 10
    pairmap = []
    for pair in jointpairs:
        candA = all_peaks[pair[0]]
        candB = all_peaks[pair[1]]
        if len(candA) == 0 or len(candB) == 0:
            pairmap.append([])
            continue
        candA = candA[0]
        candB = candB[0]
        startend = list(zip(np.linspace(candA[0], candB[0], num=mid_num), \
                                            np.linspace(candA[1], candB[1], num=mid_num)))

        vec = np.subtract(candB[:2], candA[:2])
        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
        vec = np.divide(vec, norm)
        score = 0.
        tmp = []
        for mp in map25:
            score_mid = paf[:,:,[mp[0],mp[1]]]
            vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
            vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])
            score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
            score_midpts = score_midpts.sum()
            if score < score_midpts:
                score = score_midpts
                tmp = mp
        if score > 0.5:
            pairmap.append(tmp+[score,])
        else:
            pairmap.append([])
    return pairmap
            
    