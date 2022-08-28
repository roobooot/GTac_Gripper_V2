import pandas as pd
import random

#initiate the lists:
'''
F1_x0,F1_y0,F1_z0,F1_x1,F1_y1,F1_z1,F1_x2,F1_y2,F1_z2,F1_x3,F1_y3,F1_z3,\
F1_x4,F1_y4,F1_z4,F1_x5,F1_y5,F1_z5,F1_x6,F1_y6,F1_z6,F1_x7,F1_y7,F1_z7,\
F1_x8,F1_y8,F1_z8,F1_x9,F1_y9,F1_z9,F1_x10,F1_y10,F1_z10,F1_x11,F1_y11,F1_z11,\
F1_x12,F1_y12,F1_z12,F1_x13,F1_y13,F1_z13,F1_x14,F1_y14,F1_z14,F1_x15,F1_y15,F1_z15,\
F2_x0,F2_y0,F2_z0,F2_x1,F2_y1,F2_z1,F2_x2,F2_y2,F2_z2,F2_x3,F2_y3,F2_z3,\
F2_x4,F2_y4,F2_z4,F2_x5,F2_y5,F2_z5,F2_x6,F2_y6,F2_z6,F2_x7,F2_y7,F2_z7,\
F2_x8,F2_y8,F2_z8,F2_x9,F2_y9,F2_z9,F2_x10,F2_y10,F2_z10,F2_x11,F2_y11,F2_z11,\
F2_x12,F2_y12,F2_z12,F2_x13,F2_y13,F2_z13,F2_x14,F2_y14,F2_z14,F2_x15,F2_y15,F2_z15, \
F3_x0, F3_y0, F3_z0, F3_x1, F3_y1, F3_z1, F3_x2, F3_y2, F3_z2, F3_x3, F3_y3, F3_z3, \
F3_x4, F3_y4, F3_z4, F3_x5, F3_y5, F3_z5, F3_x6, F3_y6, F3_z6, F3_x7, F3_y7, F3_z7, \
F3_x8, F3_y8, F3_z8, F3_x9, F3_y9, F3_z9, F3_x10, F3_y10, F3_z10, F3_x11, F3_y11, F3_z11, \
F3_x12, F3_y12, F3_z12, F3_x13, F3_y13, F3_z13, F3_x14, F3_y14, F3_z14, F3_x15, F3_y15, F3_z15, \
F4_x0, F4_y0, F4_z0, F4_x1, F4_y1, F4_z1, F4_x2, F4_y2, F4_z2, F4_x3, F4_y3, F4_z3, \
F4_x4, F4_y4, F4_z4, F4_x5, F4_y5, F4_z5, F4_x6, F4_y6, F4_z6, F4_x7, F4_y7, F4_z7, \
F4_x8, F4_y8, F4_z8, F4_x9, F4_y9, F4_z9, F4_x10, F4_y10, F4_z10, F4_x11, F4_y11, F4_z11, \
F4_x12, F4_y12, F4_z12, F4_x13, F4_y13, F4_z13, F4_x14, F4_y14, F4_z14, F4_x15, F4_y15, F4_z15,\
label=[]
'''

#list the list for convenience
'''
colom_list=[F1_x0,F1_y0,F1_z0,F1_x1,F1_y1,F1_z1,F1_x2,F1_y2,F1_z2,F1_x3,F1_y3,F1_z3,\
F1_x4,F1_y4,F1_z4,F1_x5,F1_y5,F1_z5,F1_x6,F1_y6,F1_z6,F1_x7,F1_y7,F1_z7,\
F1_x8,F1_y8,F1_z8,F1_x9,F1_y9,F1_z9,F1_x10,F1_y10,F1_z10,F1_x11,F1_y11,F1_z11,\
F1_x12,F1_y12,F1_z12,F1_x13,F1_y13,F1_z13,F1_x14,F1_y14,F1_z14,F1_x15,F1_y15,F1_z15,\
F2_x0,F2_y0,F2_z0,F2_x1,F2_y1,F2_z1,F2_x2,F2_y2,F2_z2,F2_x3,F2_y3,F2_z3,\
F2_x4,F2_y4,F2_z4,F2_x5,F2_y5,F2_z5,F2_x6,F2_y6,F2_z6,F2_x7,F2_y7,F2_z7,\
F2_x8,F2_y8,F2_z8,F2_x9,F2_y9,F2_z9,F2_x10,F2_y10,F2_z10,F2_x11,F2_y11,F2_z11,\
F2_x12,F2_y12,F2_z12,F2_x13,F2_y13,F2_z13,F2_x14,F2_y14,F2_z14,F2_x15,F2_y15,F2_z15, \
F3_x0, F3_y0, F3_z0, F3_x1, F3_y1, F3_z1, F3_x2, F3_y2, F3_z2, F3_x3, F3_y3, F3_z3, \
F3_x4, F3_y4, F3_z4, F3_x5, F3_y5, F3_z5, F3_x6, F3_y6, F3_z6, F3_x7, F3_y7, F3_z7, \
F3_x8, F3_y8, F3_z8, F3_x9, F3_y9, F3_z9, F3_x10, F3_y10, F3_z10, F3_x11, F3_y11, F3_z11, \
F3_x12, F3_y12, F3_z12, F3_x13, F3_y13, F3_z13, F3_x14, F3_y14, F3_z14, F3_x15, F3_y15, F3_z15, \
F4_x0, F4_y0, F4_z0, F4_x1, F4_y1, F4_z1, F4_x2, F4_y2, F4_z2, F4_x3, F4_y3, F4_z3, \
F4_x4, F4_y4, F4_z4, F4_x5, F4_y5, F4_z5, F4_x6, F4_y6, F4_z6, F4_x7, F4_y7, F4_z7, \
F4_x8, F4_y8, F4_z8, F4_x9, F4_y9, F4_z9, F4_x10, F4_y10, F4_z10, F4_x11, F4_y11, F4_z11, \
F4_x12, F4_y12, F4_z12, F4_x13, F4_y13, F4_z13, F4_x14, F4_y14, F4_z14, F4_x15, F4_y15, F4_z15,\
label]
'''
colom_list=[]

#generate data
for i in range(0,36):
    colom_list.append([])
    for j in range(1000):
        colom_list[i].append(random.uniform(1,500))

label=[]
#generate label:
for i in range(1000):
    label.append(random.randint(1,5))

#collate the data:
frame=pd.DataFrame({'F1_x0':colom_list[0], 'F1_y0':colom_list[1],'F1_z0':colom_list[2],'F1_x1':colom_list[3],'F1_y1':colom_list[4],'F1_z1':colom_list[5],'F1_x2':colom_list[6],'F1_y2':colom_list[7],'F1_z2':colom_list[8],'F1_x3':colom_list[9],'F1_y3':colom_list[10],'F1_z3':colom_list[11],\
'F1_x4':colom_list[12],'F1_y4':colom_list[13],'F1_z4':colom_list[14],'F1_x5':colom_list[15],'F1_y5':colom_list[16],'F1_z5':colom_list[17],'F1_x6':colom_list[18],'F1_y6':colom_list[19],'F1_z6':colom_list[20],'F1_x7':colom_list[21],'F1_y7':colom_list[22],'F1_z7':colom_list[23],\
'F1_x8':colom_list[24],'F1_y8':colom_list[25],'F1_z8':colom_list[26],'F1_x9':colom_list[27],'F1_y9':colom_list[28],'F1_z9':colom_list[29],'F1_x10':colom_list[30],'F1_y10':colom_list[31],'F1_z10':colom_list[32],'F1_x11':colom_list[33],'F1_y11':colom_list[34],'F1_z11':colom_list[35],\
'F1_x12':colom_list[0],'F1_y12':colom_list[27],'F1_z12':colom_list[8],'F1_x13':colom_list[27],'F1_y13':colom_list[7],'F1_z13':colom_list[14],'F1_x14':colom_list[31],'F1_y14':colom_list[16],'F1_z14':colom_list[17],'F1_x15':colom_list[26],'F1_y15':colom_list[5],'F1_z15':colom_list[7],\
'F2_x0':colom_list[24],'F2_y0':colom_list[13],'F2_z0':colom_list[13],'F2_x1':colom_list[26],'F2_y1':colom_list[18],'F2_z1':colom_list[16],'F2_x2':colom_list[31],'F2_y2':colom_list[26],'F2_z2':colom_list[7],'F2_x3':colom_list[32],'F2_y3':colom_list[18],'F2_z3':colom_list[1],\
'F2_x4':colom_list[29],'F2_y4':colom_list[23],'F2_z4':colom_list[1],'F2_x5':colom_list[3],'F2_y5':colom_list[0],'F2_z5':colom_list[24],'F2_x6':colom_list[18],'F2_y6':colom_list[26],'F2_z6':colom_list[7],'F2_x7':colom_list[8],'F2_y7':colom_list[30],'F2_z7':colom_list[32],\
'F2_x8':colom_list[29],'F2_y8':colom_list[23],'F2_z8':colom_list[13],'F2_x9':colom_list[27],'F2_y9':colom_list[13],'F2_z9':colom_list[31],'F2_x10':colom_list[14],'F2_y10':colom_list[1],'F2_z10':colom_list[18],'F2_x11':colom_list[14],'F2_y11':colom_list[16],'F2_z11':colom_list[11],\
'F2_x12':colom_list[17],'F2_y12':colom_list[23],'F2_z12':colom_list[17],'F2_x13':colom_list[3],'F2_y13':colom_list[13],'F2_z13':colom_list[1],'F2_x14':colom_list[17],'F2_y14':colom_list[32],'F2_z14':colom_list[13],'F2_x15':colom_list[8],'F2_y15':colom_list[32],'F2_z15':colom_list[7], \
'F3_x0': colom_list[26], 'F3_y0':colom_list[17],'F3_z0':colom_list[13], 'F3_x1':colom_list[26], 'F3_y1':colom_list[18],'F3_z1' :colom_list[13], 'F3_x2':colom_list[31], 'F3_y2':colom_list[18],'F3_z2': colom_list[29], 'F3_x3':colom_list[1],'F3_y3' :colom_list[8],'F3_z3' :colom_list[31], \
'F3_x4': colom_list[16],'F3_y4': colom_list[19],'F3_z4': colom_list[13],'F3_x5' :colom_list[27], 'F3_y5':colom_list[13], 'F3_z5':colom_list[26], 'F3_x6':colom_list[3],'F3_y6' :colom_list[7],'F3_z6': colom_list[13],'F3_x7' :colom_list[7],'F3_y7' :colom_list[31], 'F3_z7':colom_list[11], \
'F3_x8':colom_list[30],'F3_y8': colom_list[13],'F3_z8' :colom_list[8], 'F3_x9':colom_list[3], 'F3_y9':colom_list[17], 'F3_z9':colom_list[18], 'F3_x10':colom_list[24], 'F3_y10':colom_list[14],'F3_z10' :colom_list[16],'F3_x11' :colom_list[1],'F3_y11' :colom_list[28],'F3_z11': colom_list[5], \
'F3_x12': colom_list[5],'F3_y12': colom_list[21],'F3_z12': colom_list[13],'F3_x13': colom_list[0], 'F3_y13':colom_list[24], 'F3_z13':colom_list[26], 'F3_x14':colom_list[14], 'F3_y14':colom_list[30], 'F3_z14':colom_list[17],'F3_x15' :colom_list[29], 'F3_y15':colom_list[32], 'F3_z15':colom_list[7], \
'F4_x0':colom_list[17], 'F4_y0':colom_list[27], 'F4_z0':colom_list[3], 'F4_x1':colom_list[27], 'F4_y1':colom_list[3], 'F4_z1':colom_list[3], 'F4_x2':colom_list[14], 'F4_y2':colom_list[26], 'F4_z2':colom_list[17], 'F4_x3':colom_list[11], 'F4_y3':colom_list[32], 'F4_z3':colom_list[5], \
'F4_x4': colom_list[0],'F4_y4' :colom_list[23],'F4_z4' :colom_list[24],'F4_x5': colom_list[5], 'F4_y5':colom_list[16], 'F4_z5':colom_list[7],'F4_x6' :colom_list[16], 'F4_y6':colom_list[8], 'F4_z6':colom_list[31], 'F4_x7':colom_list[30], 'F4_y7':colom_list[14], 'F4_z7':colom_list[5], \
'F4_x8':colom_list[28], 'F4_y8':colom_list[8], 'F4_z8':colom_list[27], 'F4_x9':colom_list[29], 'F4_y9':colom_list[0], 'F4_z9':colom_list[0],'F4_x10': colom_list[24], 'F4_y10':colom_list[30], 'F4_z10':colom_list[24],'F4_x11': colom_list[14],'F4_y11': colom_list[8],'F4_z11': colom_list[28], \
'F4_x12':colom_list[16], 'F4_y12':colom_list[29], 'F4_z12':colom_list[17], 'F4_x13':colom_list[29],'F4_y13': colom_list[24],'F4_z13': colom_list[16],'F4_x14': colom_list[8], 'F4_y14':colom_list[30], 'F4_z14':colom_list[32],'F4_x15': colom_list[31], 'F4_y15':colom_list[32],'F4_z15': colom_list[28],\
'F1_x0':colom_list[0], 'F1_y0':colom_list[1],'F1_z0':colom_list[2],'F1_x1':colom_list[3],'F1_y1':colom_list[4],'F1_z1':colom_list[5],'F1_x2':colom_list[6],'F1_y2':colom_list[7],'F1_z2':colom_list[8],'F1_x3':colom_list[9],'F1_y3':colom_list[10],'F1_z3':colom_list[11],\
'F1_x4':colom_list[12],'F1_y4':colom_list[13],'F1_z4':colom_list[14],'F1_x5':colom_list[15],'F1_y5':colom_list[16],'F1_z5':colom_list[17],'F1_x6':colom_list[18],'F1_y6':colom_list[19],'F1_z6':colom_list[20],'F1_x7':colom_list[21],'F1_y7':colom_list[22],'F1_z7':colom_list[23],\
'F1_x8':colom_list[24],'F1_y8':colom_list[25],'F1_z8':colom_list[26],'F1_x9':colom_list[27],'F1_y9':colom_list[28],'F1_z9':colom_list[29],'F1_x10':colom_list[30],'F1_y10':colom_list[31],'F1_z10':colom_list[32],'F1_x11':colom_list[33],'F1_y11':colom_list[34],'F1_z11':colom_list[35],\
'F1_x12':colom_list[0],'F1_y12':colom_list[27],'F1_z12':colom_list[8],'F1_x13':colom_list[27],'F1_y13':colom_list[7],'F1_z13':colom_list[14],'F1_x14':colom_list[31],'F1_y14':colom_list[16],'F1_z14':colom_list[17],'F1_x15':colom_list[26],'F1_y15':colom_list[5],'F1_z15':colom_list[7],\
'F2_x0':colom_list[24],'F2_y0':colom_list[13],'F2_z0':colom_list[13],'F2_x1':colom_list[26],'F2_y1':colom_list[18],'F2_z1':colom_list[16],'F2_x2':colom_list[31],'F2_y2':colom_list[26],'F2_z2':colom_list[7],'F2_x3':colom_list[32],'F2_y3':colom_list[18],'F2_z3':colom_list[1],\
'F2_x4':colom_list[29],'F2_y4':colom_list[23],'F2_z4':colom_list[1],'F2_x5':colom_list[3],'F2_y5':colom_list[0],'F2_z5':colom_list[24],'F2_x6':colom_list[18],'F2_y6':colom_list[26],'F2_z6':colom_list[7],'F2_x7':colom_list[8],'F2_y7':colom_list[30],'F2_z7':colom_list[32],\
'F2_x8':colom_list[29],'F2_y8':colom_list[23],'F2_z8':colom_list[13],'F2_x9':colom_list[27],'F2_y9':colom_list[13],'F2_z9':colom_list[31],'F2_x10':colom_list[14],'F2_y10':colom_list[1],'F2_z10':colom_list[18],'F2_x11':colom_list[14],'F2_y11':colom_list[16],'F2_z11':colom_list[11],\
'F2_x12':colom_list[17],'F2_y12':colom_list[23],'F2_z12':colom_list[17],'F2_x13':colom_list[3],'F2_y13':colom_list[13],'F2_z13':colom_list[1],'F2_x14':colom_list[17],'F2_y14':colom_list[32],'F2_z14':colom_list[13],'F2_x15':colom_list[8],'F2_y15':colom_list[32],'F2_z15':colom_list[7], \
'F3_x0': colom_list[26], 'F3_y0':colom_list[17],'F3_z0':colom_list[13], 'F3_x1':colom_list[26], 'F3_y1':colom_list[18],'F3_z1' :colom_list[13], 'F3_x2':colom_list[31], 'F3_y2':colom_list[18],'F3_z2': colom_list[29], 'F3_x3':colom_list[1],'F3_y3' :colom_list[8],'F3_z3' :colom_list[31], \
'F3_x4': colom_list[16],'F3_y4': colom_list[19],'F3_z4': colom_list[13],'F3_x5' :colom_list[27], 'F3_y5':colom_list[13], 'F3_z5':colom_list[26], 'F3_x6':colom_list[3],'F3_y6' :colom_list[7],'F3_z6': colom_list[13],'F3_x7' :colom_list[7],'F3_y7' :colom_list[31], 'F3_z7':colom_list[11], \
'F3_x8':colom_list[30],'F3_y8': colom_list[13],'F3_z8' :colom_list[8], 'F3_x9':colom_list[3], 'F3_y9':colom_list[17], 'F3_z9':colom_list[18], 'F3_x10':colom_list[24], 'F3_y10':colom_list[14],'F3_z10' :colom_list[16],'F3_x11' :colom_list[1],'F3_y11' :colom_list[28],'F3_z11': colom_list[5], \
'F3_x12': colom_list[5],'F3_y12': colom_list[21],'F3_z12': colom_list[13],'F3_x13': colom_list[0], 'F3_y13':colom_list[24], 'F3_z13':colom_list[26], 'F3_x14':colom_list[14], 'F3_y14':colom_list[30], 'F3_z14':colom_list[17],'F3_x15' :colom_list[29], 'F3_y15':colom_list[32], 'F3_z15':colom_list[7], \
'F4_x0':colom_list[17], 'F4_y0':colom_list[27], 'F4_z0':colom_list[3], 'F4_x1':colom_list[27], 'F4_y1':colom_list[3], 'F4_z1':colom_list[3], 'F4_x2':colom_list[14], 'F4_y2':colom_list[26], 'F4_z2':colom_list[17], 'F4_x3':colom_list[11], 'F4_y3':colom_list[32], 'F4_z3':colom_list[5], \
'F4_x4': colom_list[0],'F4_y4' :colom_list[23],'F4_z4' :colom_list[24],'F4_x5': colom_list[5], 'F4_y5':colom_list[16], 'F4_z5':colom_list[7],'F4_x6' :colom_list[16], 'F4_y6':colom_list[8], 'F4_z6':colom_list[31], 'F4_x7':colom_list[30], 'F4_y7':colom_list[14], 'F4_z7':colom_list[5], \
'F4_x8':colom_list[28], 'F4_y8':colom_list[8], 'F4_z8':colom_list[27], 'F4_x9':colom_list[29], 'F4_y9':colom_list[0], 'F4_z9':colom_list[0],'F4_x10': colom_list[24], 'F4_y10':colom_list[30], 'F4_z10':colom_list[24],'F4_x11': colom_list[14],'F4_y11': colom_list[8],'F4_z11': colom_list[28], \
'F4_x12':colom_list[16], 'F4_y12':colom_list[29], 'F4_z12':colom_list[17], 'F4_x13':colom_list[29],'F4_y13': colom_list[24],'F4_z13': colom_list[16],'F4_x14': colom_list[8], 'F4_y14':colom_list[30], 'F4_z14':colom_list[32],'F4_x15': colom_list[31], 'F4_y15':colom_list[32],'F4_z15': colom_list[28],\

'F11_x0':colom_list[0], 'F11_y0':colom_list[1],'F11_z0':colom_list[2],'F11_x1':colom_list[3],'F11_y1':colom_list[4],'F11_z1':colom_list[5],'F11_x2':colom_list[6],'F11_y2':colom_list[7],'F11_z2':colom_list[8],'F11_x3':colom_list[9],'F11_y3':colom_list[10],'F11_z3':colom_list[11],\
'F11_x4':colom_list[12],'F11_y4':colom_list[13],'F11_z4':colom_list[14],'F11_x5':colom_list[15],'F11_y5':colom_list[16],'F11_z5':colom_list[17],'F11_x6':colom_list[18],'F11_y6':colom_list[19],'F11_z6':colom_list[20],'F11_x7':colom_list[21],'F11_y7':colom_list[22],'F11_z7':colom_list[23],\
'F11_x8':colom_list[24],'F11_y8':colom_list[25],'F11_z8':colom_list[26],'F11_x9':colom_list[27],'F11_y9':colom_list[28],'F11_z9':colom_list[29],'F11_x10':colom_list[30],'F11_y10':colom_list[31],'F11_z10':colom_list[32],'F11_x11':colom_list[33],'F11_y11':colom_list[34],'F11_z11':colom_list[35],\
'F11_x12':colom_list[0],'F11_y12':colom_list[27],'F11_z12':colom_list[8],'F11_x13':colom_list[27],'F11_y13':colom_list[7],'F11_z13':colom_list[14],'F11_x14':colom_list[31],'F11_y14':colom_list[16],'F11_z14':colom_list[17],'F11_x15':colom_list[26],'F11_y15':colom_list[5],'F11_z15':colom_list[7],\
'F21_x0':colom_list[24],'F21_y0':colom_list[13],'F21_z0':colom_list[13],'F21_x1':colom_list[26],'F21_y1':colom_list[18],'F21_z1':colom_list[16],'F21_x2':colom_list[31],'F21_y2':colom_list[26],'F21_z2':colom_list[7],'F21_x3':colom_list[32],'F21_y3':colom_list[18],'F21_z3':colom_list[1],\
'F21_x4':colom_list[29],'F21_y4':colom_list[23],'F21_z4':colom_list[1],'F21_x5':colom_list[3],'F21_y5':colom_list[0],'F21_z5':colom_list[24],'F21_x6':colom_list[18],'F21_y6':colom_list[26],'F21_z6':colom_list[7],'F21_x7':colom_list[8],'F21_y7':colom_list[30],'F21_z7':colom_list[32],\
'F21_x8':colom_list[29],'F21_y8':colom_list[23],'F21_z8':colom_list[13],'F21_x9':colom_list[27],'F21_y9':colom_list[13],'F21_z9':colom_list[31],'F21_x10':colom_list[14],'F21_y10':colom_list[1],'F21_z10':colom_list[18],'F21_x11':colom_list[14],'F21_y11':colom_list[16],'F21_z11':colom_list[11],\
'F21_x12':colom_list[17],'F21_y12':colom_list[23],'F21_z12':colom_list[17],'F21_x13':colom_list[3],'F21_y13':colom_list[13],'F21_z13':colom_list[1],'F21_x14':colom_list[17],'F21_y14':colom_list[32],'F21_z14':colom_list[13],'F21_x15':colom_list[8],'F21_y15':colom_list[32],'F21_z15':colom_list[7], \
'F31_x0': colom_list[26], 'F31_y0':colom_list[17],'F31_z0':colom_list[13], 'F31_x1':colom_list[26], 'F31_y1':colom_list[18],'F31_z1' :colom_list[13], 'F31_x2':colom_list[31], 'F31_y2':colom_list[18],'F31_z2': colom_list[29], 'F31_x3':colom_list[1],'F31_y3' :colom_list[8],'F31_z3' :colom_list[31], \
'F31_x4': colom_list[16],'F31_y4': colom_list[19],'F31_z4': colom_list[13],'F31_x5' :colom_list[27], 'F31_y5':colom_list[13], 'F31_z5':colom_list[26], 'F31_x6':colom_list[3],'F31_y6' :colom_list[7],'F31_z6': colom_list[13],'F31_x7' :colom_list[7],'F31_y7' :colom_list[31], 'F31_z7':colom_list[11], \
'F31_x8':colom_list[30],'F31_y8': colom_list[13],'F31_z8' :colom_list[8], 'F31_x9':colom_list[3], 'F31_y9':colom_list[17], 'F31_z9':colom_list[18], 'F31_x10':colom_list[24], 'F31_y10':colom_list[14],'F31_z10' :colom_list[16],'F31_x11' :colom_list[1],'F31_y11' :colom_list[28],'F31_z11': colom_list[5], \
'F31_x12': colom_list[5],'F31_y12': colom_list[21],'F31_z12': colom_list[13],'F31_x13': colom_list[0], 'F31_y13':colom_list[24], 'F31_z13':colom_list[26], 'F31_x14':colom_list[14], 'F31_y14':colom_list[30], 'F31_z14':colom_list[17],'F31_x15' :colom_list[29], 'F31_y15':colom_list[32], 'F31_z15':colom_list[7], \
'F41_x0':colom_list[17], 'F41_y0':colom_list[27], 'F41_z0':colom_list[3], 'F41_x1':colom_list[27], 'F41_y1':colom_list[3], 'F41_z1':colom_list[3], 'F41_x2':colom_list[14], 'F41_y2':colom_list[26], 'F41_z2':colom_list[17], 'F41_x3':colom_list[11], 'F41_y3':colom_list[32], 'F41_z3':colom_list[5], \
'F41_x4': colom_list[0],'F41_y4' :colom_list[23],'F41_z4' :colom_list[24],'F41_x5': colom_list[5], 'F41_y5':colom_list[16], 'F41_z5':colom_list[7],'F41_x6' :colom_list[16], 'F41_y6':colom_list[8], 'F41_z6':colom_list[31], 'F41_x7':colom_list[30], 'F41_y7':colom_list[14], 'F41_z7':colom_list[5], \
'F41_x8':colom_list[28], 'F41_y8':colom_list[8], 'F41_z8':colom_list[27], 'F41_x9':colom_list[29], 'F41_y9':colom_list[0], 'F41_z9':colom_list[0],'F41_x10': colom_list[24], 'F41_y10':colom_list[30], 'F41_z10':colom_list[24],'F41_x11': colom_list[14],'F41_y11': colom_list[8],'F41_z11': colom_list[28], \
'F41_x12':colom_list[16], 'F41_y12':colom_list[29], 'F41_z12':colom_list[17], 'F41_x13':colom_list[29],'F41_y13': colom_list[24],'F41_z13': colom_list[16],'F41_x14': colom_list[8], 'F41_y14':colom_list[30], 'F41_z14':colom_list[32],'F41_x15': colom_list[31], 'F41_y15':colom_list[32],'F41_z15': colom_list[28],\

'F12_x0':colom_list[0], 'F12_y0':colom_list[1],'F12_z0':colom_list[2],'F12_x1':colom_list[3],'F12_y1':colom_list[4],'F12_z1':colom_list[5],'F12_x2':colom_list[6],'F12_y2':colom_list[7],'F12_z2':colom_list[8],'F12_x3':colom_list[9],'F12_y3':colom_list[10],'F12_z3':colom_list[11],\
'F12_x4':colom_list[12],'F12_y4':colom_list[13],'F12_z4':colom_list[14],'F12_x5':colom_list[15],'F12_y5':colom_list[16],'F12_z5':colom_list[17],'F12_x6':colom_list[18],'F12_y6':colom_list[19],'F12_z6':colom_list[20],'F12_x7':colom_list[21],'F12_y7':colom_list[22],'F12_z7':colom_list[23],\
'F12_x8':colom_list[24],'F12_y8':colom_list[25],'F12_z8':colom_list[26],'F12_x9':colom_list[27],'F12_y9':colom_list[28],'F12_z9':colom_list[29],'F12_x10':colom_list[30],'F12_y10':colom_list[31],'F12_z10':colom_list[32],'F12_x11':colom_list[33],'F12_y11':colom_list[34],'F12_z11':colom_list[35],\
'F12_x12':colom_list[0],'F12_y12':colom_list[27],'F12_z12':colom_list[8],'F12_x13':colom_list[27],'F12_y13':colom_list[7],'F12_z13':colom_list[14],'F12_x14':colom_list[31],'F12_y14':colom_list[16],'F12_z14':colom_list[17],'F12_x15':colom_list[26],'F12_y15':colom_list[5],'F12_z15':colom_list[7],\
'F22_x0':colom_list[24],'F22_y0':colom_list[13],'F22_z0':colom_list[13],'F22_x1':colom_list[26],'F22_y1':colom_list[18],'F22_z1':colom_list[16],'F22_x2':colom_list[31],'F22_y2':colom_list[26],'F22_z2':colom_list[7],'F22_x3':colom_list[32],'F22_y3':colom_list[18],'F22_z3':colom_list[1],\
'F22_x4':colom_list[29],'F22_y4':colom_list[23],'F22_z4':colom_list[1],'F22_x5':colom_list[3],'F22_y5':colom_list[0],'F22_z5':colom_list[24],'F22_x6':colom_list[18],'F22_y6':colom_list[26],'F22_z6':colom_list[7],'F22_x7':colom_list[8],'F22_y7':colom_list[30],'F22_z7':colom_list[32],\
'F22_x8':colom_list[29],'F22_y8':colom_list[23],'F22_z8':colom_list[13],'F22_x9':colom_list[27],'F22_y9':colom_list[13],'F22_z9':colom_list[31],'F22_x10':colom_list[14],'F22_y10':colom_list[1],'F22_z10':colom_list[18],'F22_x11':colom_list[14],'F22_y11':colom_list[16],'F22_z11':colom_list[11],\
'F22_x12':colom_list[17],'F22_y12':colom_list[23],'F22_z12':colom_list[17],'F22_x13':colom_list[3],'F22_y13':colom_list[13],'F22_z13':colom_list[1],'F22_x14':colom_list[17],'F22_y14':colom_list[32],'F22_z14':colom_list[13],'F22_x15':colom_list[8],'F22_y15':colom_list[32],'F22_z15':colom_list[7], \
'F32_x0': colom_list[26], 'F32_y0':colom_list[17],'F32_z0':colom_list[13], 'F32_x1':colom_list[26], 'F32_y1':colom_list[18],'F32_z1' :colom_list[13], 'F32_x2':colom_list[31], 'F32_y2':colom_list[18],'F32_z2': colom_list[29], 'F32_x3':colom_list[1],'F32_y3' :colom_list[8],'F32_z3' :colom_list[31], \
'F32_x4': colom_list[16],'F32_y4': colom_list[19],'F32_z4': colom_list[13],'F32_x5' :colom_list[27], 'F32_y5':colom_list[13], 'F32_z5':colom_list[26], 'F32_x6':colom_list[3],'F32_y6' :colom_list[7],'F32_z6': colom_list[13],'F32_x7' :colom_list[7],'F32_y7' :colom_list[31], 'F32_z7':colom_list[11], \
'F32_x8':colom_list[30],'F32_y8': colom_list[13],'F32_z8' :colom_list[8], 'F32_x9':colom_list[3], 'F32_y9':colom_list[17], 'F32_z9':colom_list[18], 'F32_x10':colom_list[24], 'F32_y10':colom_list[14],'F32_z10' :colom_list[16],'F32_x11' :colom_list[1],'F32_y11' :colom_list[28],'F32_z11': colom_list[5], \
'F32_x12': colom_list[5],'F32_y12': colom_list[21],'F32_z12': colom_list[13],'F32_x13': colom_list[0], 'F32_y13':colom_list[24], 'F32_z13':colom_list[26], 'F32_x14':colom_list[14], 'F32_y14':colom_list[30], 'F32_z14':colom_list[17],'F32_x15' :colom_list[29], 'F32_y15':colom_list[32], 'F32_z15':colom_list[7], \
'F42_x0':colom_list[17], 'F42_y0':colom_list[27], 'F42_z0':colom_list[3], 'F42_x1':colom_list[27], 'F42_y1':colom_list[3], 'F42_z1':colom_list[3], 'F42_x2':colom_list[14], 'F42_y2':colom_list[26], 'F42_z2':colom_list[17], 'F42_x3':colom_list[11], 'F42_y3':colom_list[32], 'F42_z3':colom_list[5], \
'F42_x4': colom_list[0],'F42_y4' :colom_list[23],'F42_z4' :colom_list[24],'F42_x5': colom_list[5], 'F42_y5':colom_list[16], 'F42_z5':colom_list[7],'F42_x6' :colom_list[16], 'F42_y6':colom_list[8], 'F42_z6':colom_list[31], 'F42_x7':colom_list[30], 'F42_y7':colom_list[14], 'F42_z7':colom_list[5], \
'F42_x8':colom_list[28], 'F42_y8':colom_list[8], 'F42_z8':colom_list[27], 'F42_x9':colom_list[29], 'F42_y9':colom_list[0], 'F42_z9':colom_list[0],'F42_x10': colom_list[24], 'F42_y10':colom_list[30], 'F42_z10':colom_list[24],'F42_x11': colom_list[14],'F42_y11': colom_list[8],'F42_z11': colom_list[28], \
'F42_x12':colom_list[16], 'F42_y12':colom_list[29], 'F42_z12':colom_list[17], 'F42_x13':colom_list[29],'F42_y13': colom_list[24],'F42_z13': colom_list[16],'F42_x14': colom_list[8], 'F42_y14':colom_list[30], 'F42_z14':colom_list[32],'F42_x15': colom_list[31], 'F42_y15':colom_list[32],'F42_z15': colom_list[28],\
'label':label})

#create csv file
frame.to_csv("C:\\Users\\Lenovo\\Desktop\\tactile-gcn-master\\data\\biotacsp\\raw\\peudo1.csv",index=False,sep=',')


'''
import argparse
import copy
import math
import os
import pickle
import time

import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import gtac_config
import numpy as np
import ntpath
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import svm
from scipy import signal
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold


def find_sec_data(data_frame_array, finger, sec, a=0.1):
    # Input: data frame
    # Output: the GTac data in one finger section, shape = 19 [f16,s3]
    sec_data = []
    for i in range(gtac_config.MAT_NUM):  # MAT_NUM -> 16: there 4*4 sensing points on FA-I layer
        r = i // 4
        c = i % 4
        index, value = gtac_data.find_FAI_value(data_frame_array, finger, sec, r, c)
        sec_data.append(value)
    FA_sum = np.sum(sec_data)
    mag_all, _ = gtac_data.find_SAII(data_frame_array, finger, sec)
    g = math.sqrt(mag_all[0] * mag_all[0] + mag_all[1] * mag_all[1] + (a * mag_all[2] + (1 - a) * FA_sum) * (
            a * mag_all[2] + (1 - a) * FA_sum))
    g = round(g, 2)
    for m in mag_all:
        sec_data.append(m)
    return sec_data, g

'''
