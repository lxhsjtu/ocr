import math
import json
import cv2
import xlrd
import time
import os

def loaddict():
    xlsxpath = '..//dataset//words_index.xlsx'
    file = xlrd.open_workbook(xlsxpath)
    char_list=[]
    try:
        sh = file.sheet_by_name(u"Sheet1")
    except:
        print("no sheet")
    dict = {}
    for i in range(0, 6866):
        char = sh.cell_value(i, 0)
        index = int(sh.cell(i, 1).value)
        dict[char] = index+2
        char_list.append(char)
    return dict,char_list