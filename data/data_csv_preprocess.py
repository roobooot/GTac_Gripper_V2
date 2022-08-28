import gtac_config
import math
import pandas as pd
import numpy as np
import ntpath
import copy
import os
import random
from sklearn.utils import shuffle

#csv_name='C:\\Users\\Lenovo\\Desktop\\tactile-gcn-12features\\data\\biotacsp\\raw\\winebottle_test_260_GTAC_20220822_125941.csv'

class data_preprocess_csv():

    def __init__(self,ir):
        self.csv_name=ir#'C:\\Users\\Lenovo\\Desktop\\tactile-gcn-12features\\data\\biotacsp\\raw\\winebottle_test_260_GTAC_20220822_125941.csv'

        self.init_csv()


        self.normal_force=[[]for i in range(48)]#np.zeros(48)
        self.shear_x=[[] for i in range(3)]
        self.shear_y=[[] for i in range(3)]
        self.shear_z=[[] for i in range(3)]
    '''
    def variate_dir(self,dir):
        self.csv_name=dir
        return self.csv_name
    '''


    def init_csv(self, cut_pos=-100):
        self.csv_data = pd.read_csv(self.csv_name)
        # print(f'before{self.csv_data.head(5)}')
        self.csv_data = self.csv_data.iloc[cut_pos:, 1:]
        # print(self.csv_data.head(5))



    @staticmethod
    def find_FAI_index(finger, sec, r, c):
        # the FAI index exclude the mag data
        # the overall index is "output"+NUM_MAG
        index = sec * 4 * 20 + finger * 4 + \
                gtac_config.MAT_ORIENT_COL[finger, sec, r, c] * 20 + \
                gtac_config.MAT_ORIENT_ROW[finger, sec, r, c]
        return index + gtac_config.MAG_NUM

    @staticmethod
    def find_FAI_value(data_frame_array, finger, sec, r, c):
        index = data_preprocess_csv.find_FAI_index(finger, sec, r, c)
        return index, data_frame_array[index]

    @staticmethod
    def find_SAII(data_frame_array, finger, sec):
        # print('SAII_data: finger {}, sec {}'.format(finger, sec))
        tri_index = finger * 9 + (2 - sec) * 3
        mag_x = data_frame_array[tri_index]
        mag_y = data_frame_array[tri_index + 1]
        mag_z = data_frame_array[tri_index + 2]
        SAII_scaler = math.sqrt(mag_x * mag_x + mag_y * mag_y + mag_z * mag_z)
        # print('Find_SAII: {} scalar: {}'.format([mag_x, mag_y, mag_z], SAII_scaler))
        return [mag_x, mag_y, mag_z], SAII_scaler

    @staticmethod
    def find_SAII_index(finger, sec):
        tri_index = finger * 9 + (2 - sec) * 3
        return tri_index, tri_index + 1, tri_index + 2

    #@staticmethod
    #find the normal_force(exclude the normal shear force) of certain node
    #input:finger,section,r,c
    #output:colum_list of the normal force [nx1]
    def find_normal_force(self,finger):
        #normal_force=np.zeros(16)
        for section in range(3):
            for i in range(16):
                r=i//4
                c=i%4
                index_normal=data_preprocess_csv.find_FAI_index(finger,section,r,c)
                #self.normal_force[16*section+i]= np.copy(self.csv_data.iloc([index_normal]).astype(np.int, copy=False))
                #test=[]
                #test=pd.read_csv(self.csv_name, usecols =[index_normal])
                #print(test)
                #self.normal_force[16 * section + i] = np.copy(pd.read_csv(self.csv_name, usecols =[index_normal],skiprows=[0]).astype(np.int, copy=False))
                for j in range(len(self.csv_data)):
                    sample_ = self.csv_data.iloc[j]
                    num=sample_.iloc[index_normal]
                    self.normal_force[16*section+i].append(sample_.iloc[index_normal])
            #self.normal_force[16 * section + i] = np.copy(self.csv_data.iloc([index_normal]).astype(np.int, copy=False))
            #print(self.normal_force)

        return self.normal_force

    # find the normal_force(exclude the normal shear force) of certain node
    # input:finger,section,r,c
    # output:colum_list of the normal force [nx1]
    def find_shear_force(self,finger):
        for section in range(3):
            shear_index_x,shear_index_y,shear_index_z=data_preprocess_csv.find_SAII_index(finger,section)
            for j in range(len(self.csv_data)):
                sample_ = self.csv_data.iloc[j]
                num = sample_.iloc[shear_index_x]
                self.shear_x[section].append(sample_.iloc[shear_index_x]/16)
                self.shear_y[section].append(sample_.iloc[shear_index_y]/16)
                self.shear_z[section].append(sample_.iloc[shear_index_z]/16)
                # self.normal_force[16 * section + i] = np.copy(self.csv_data.iloc([index_normal]).astype(np.int, copy=False))
        #print(self.shear_x)
        return self.shear_x,self.shear_y,self.shear_z

    @staticmethod
    def get_label_from_filename(filename, skip=0):
        filename = ntpath.basename(filename)
        filename = filename[skip:]
        label = ''
        #orient = ''
        for x, c in enumerate(filename):
            if not c == '_':
                label += c
            elif not filename[x + 1].isdigit():
                label += c
            elif filename[x + 1].isdigit():
                break
        l_label = len(label)
        #print(f'label in spring is {label}')
        '''
        for i in range(l_label):
            if filename[l_label - 1 - i].isdigit():
                orient += filename[l_label - 1 - i]
                label = label[:l_label - 3 - i]
            else:
                break
        print(f'orient is {orient[::-1]}')
        orient = int(orient[::-1])
        '''
        obj = label

        label = copy.copy(gtac_config.objects_dict[label])
        #label.append(orient)
        print(f'label extration: {label}')
        return label,obj

file_dir=''
file_name=''

def create_csv_raws(data_dir,Raw_prime_Folder_path,augment=False):
    file_num=0
    for filename in os.listdir(data_dir):
        if os.path.splitext(filename)[-1] == '.csv':
            file_num+=1

            #assigne the data and label values to the csv file
            #data_preprocess_csv().csv_name=os.path.join(data_dir,filename)
            mid_file_name=os.path.join(data_dir,filename)
#            mid_file_name_t=data_preprocess_csv().variate_dir(mid_file_name)
            mid_file_name_t = data_preprocess_csv(mid_file_name)

            normal_prime0=data_preprocess_csv(mid_file_name).find_normal_force(finger=0)
            shear_data_x0,shear_data_y0,shear_data_z0=data_preprocess_csv(mid_file_name).find_shear_force(finger=4)
            for i in range(len(normal_prime0)):
                for j in range(len(shear_data_z0[1])):
                    normal_prime0[i][j]=normal_prime0[i][j]+shear_data_z0[2][j]
            normal_prime1=data_preprocess_csv(mid_file_name).find_normal_force(finger=1)
            shear_data_x1,shear_data_y1,shear_data_z1=data_preprocess_csv(mid_file_name).find_shear_force(finger=1)
            for i in range(len(normal_prime1)):
                for j in range(len(shear_data_z0[1])):
                    normal_prime1[i][j]=normal_prime1[i][j]+shear_data_z1[2][j]
            normal_prime2=data_preprocess_csv(mid_file_name).find_normal_force(finger=2)
            shear_data_x2,shear_data_y2,shear_data_z2=data_preprocess_csv(mid_file_name).find_shear_force(finger=2)
            for i in range(len(normal_prime2)):
                for j in range(len(shear_data_z0[1])):
                    normal_prime2[i][j]=normal_prime2[i][j]+shear_data_z2[2][j]
            normal_prime3=data_preprocess_csv(mid_file_name).find_normal_force(finger=3)
            shear_data_x3,shear_data_y3,shear_data_z3=data_preprocess_csv(mid_file_name).find_shear_force(finger=3)
            for i in range(len(normal_prime3)):
                for j in range(len(shear_data_z3[1])):
                    normal_prime3[i][j]=normal_prime3[i][j]+shear_data_z3[2][j]

            label_prime,_=data_preprocess_csv(mid_file_name).get_label_from_filename(filename=data_preprocess_csv(mid_file_name).csv_name)
            label=[]
            for i in range(4*len(shear_data_x2[0])):#times 4 to get the label size with the same length
                label.append(label_prime[0])
            #print(label)
            ######




            #collate the data:
            frame=pd.DataFrame({
            'F1_x0':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0], 'F1_x1':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0], 'F1_x2':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0],'F1_x3':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0],
            'F1_x4':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0], 'F1_x5':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0], 'F1_x6':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0],'F1_x7':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0],
            'F1_x8':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0], 'F1_x9':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0], 'F1_x10':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0],'F1_x11':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0],
            'F1_x12':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0], 'F1_x13':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0], 'F1_x14':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0],'F1_x15':shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0],

            'F1_x16':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1], 'F1_x17':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1], 'F1_x18':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1],'F1_x19':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1],
            'F1_x20':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1], 'F1_x21':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1], 'F1_x22':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1],'F1_x23':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1],
            'F1_x24':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1], 'F1_x25':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1], 'F1_x26':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1],'F1_x27':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1],
            'F1_x28':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1], 'F1_x29':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1], 'F1_x30':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1],'F1_x31':shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1],

            'F1_x32':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2], 'F1_x33':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2], 'F1_x34':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2],'F1_x35':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2],
            'F1_x36':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2], 'F1_x37':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2], 'F1_x38':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2],'F1_x39':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2],
            'F1_x40':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2], 'F1_x41':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2], 'F1_x42':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2],'F1_x43':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2],
            'F1_x44':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2], 'F1_x45':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2], 'F1_x46':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2],'F1_x47':shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2],

            'F1_y0':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0], 'F1_y1':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0], 'F1_y2':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0],'F1_y3':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0],
            'F1_y4':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0], 'F1_y5':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0], 'F1_y6':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0],'F1_y7':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0],
            'F1_y8':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0], 'F1_y9':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0], 'F1_y10':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0],'F1_y11':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0],
            'F1_y12':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0], 'F1_y13':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0], 'F1_y14':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0],'F1_y15':shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0],

            'F1_y16':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1], 'F1_y17':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1], 'F1_y18':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1],'F1_y19':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1],
            'F1_y20':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1], 'F1_y21':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1], 'F1_y22':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1],'F1_y23':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1],
            'F1_y24':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1], 'F1_y25':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1], 'F1_y26':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1],'F1_y27':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1],
            'F1_y28':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1], 'F1_y29':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1], 'F1_y30':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1],'F1_y31':shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1],

            'F1_y32':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2], 'F1_y33':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2], 'F1_y34':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2],'F1_y35':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2],
            'F1_y36':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2], 'F1_y37':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2], 'F1_y38':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2],'F1_y39':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2],
            'F1_y40':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2], 'F1_y41':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2], 'F1_y42':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2],'F1_y43':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2],
            'F1_y44':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2], 'F1_y45':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2], 'F1_y46':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2],'F1_y47':shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2],



            'F1_z0':normal_prime0[0]+normal_prime1[0]+normal_prime2[0]+normal_prime3[0], 'F1_z1':normal_prime0[1]+normal_prime1[1]+normal_prime2[1]+normal_prime3[1], 'F1_z2':normal_prime0[2]+normal_prime1[2]+normal_prime2[2]+normal_prime3[2],'F1_z3':normal_prime0[3]+normal_prime1[3]+normal_prime2[3]+normal_prime3[3],
            'F1_z4':normal_prime0[4]+normal_prime1[4]+normal_prime2[4]+normal_prime3[4], 'F1_z5':normal_prime0[5]+normal_prime1[5]+normal_prime2[5]+normal_prime3[5], 'F1_z6':normal_prime0[6]+normal_prime1[6]+normal_prime2[6]+normal_prime3[6],'F1_z7':normal_prime0[7]+normal_prime1[7]+normal_prime2[7]+normal_prime3[7],
            'F1_z8':normal_prime0[8]+normal_prime1[8]+normal_prime2[8]+normal_prime3[8], 'F1_z9':normal_prime0[9]+normal_prime1[9]+normal_prime2[9]+normal_prime3[9], 'F1_z10':normal_prime0[10]+normal_prime1[10]+normal_prime2[10]+normal_prime3[10],'F1_z11':normal_prime0[11]+normal_prime1[11]+normal_prime2[11]+normal_prime3[11],
            'F1_z12':normal_prime0[12]+normal_prime1[12]+normal_prime2[12]+normal_prime3[12], 'F1_z13':normal_prime0[13]+normal_prime1[13]+normal_prime2[13]+normal_prime3[13], 'F1_z14':normal_prime0[14]+normal_prime1[14]+normal_prime2[14]+normal_prime3[14],'F1_z15':normal_prime0[15]+normal_prime1[15]+normal_prime2[15]+normal_prime3[15],
            'F1_z16':normal_prime0[16]+normal_prime1[16]+normal_prime2[16]+normal_prime3[16], 'F1_z17':normal_prime0[17]+normal_prime1[17]+normal_prime2[17]+normal_prime3[17], 'F1_z18':normal_prime0[18]+normal_prime1[18]+normal_prime2[18]+normal_prime3[18],'F1_z19':normal_prime0[19]+normal_prime1[10]+normal_prime2[19]+normal_prime3[19],
            'F1_z20':normal_prime0[20]+normal_prime1[20]+normal_prime2[20]+normal_prime3[20], 'F1_z21':normal_prime0[21]+normal_prime1[21]+normal_prime2[21]+normal_prime3[21], 'F1_z22':normal_prime0[22]+normal_prime1[22]+normal_prime2[22]+normal_prime3[22],'F1_z23':normal_prime0[23]+normal_prime1[23]+normal_prime2[23]+normal_prime3[23],
            'F1_z24':normal_prime0[24]+normal_prime1[24]+normal_prime2[24]+normal_prime3[24], 'F1_z25':normal_prime0[25]+normal_prime1[25]+normal_prime2[25]+normal_prime3[25], 'F1_z26':normal_prime0[26]+normal_prime1[26]+normal_prime2[26]+normal_prime3[26],'F1_z27':normal_prime0[27]+normal_prime1[27]+normal_prime2[27]+normal_prime3[27],
            'F1_z28':normal_prime0[28]+normal_prime1[28]+normal_prime2[28]+normal_prime3[28], 'F1_z29':normal_prime0[29]+normal_prime1[29]+normal_prime2[29]+normal_prime3[29], 'F1_z30':normal_prime0[30]+normal_prime1[30]+normal_prime2[30]+normal_prime3[30],'F1_z31':normal_prime0[31]+normal_prime1[31]+normal_prime2[31]+normal_prime3[31],
            'F1_z32':normal_prime0[32]+normal_prime1[32]+normal_prime2[32]+normal_prime3[32], 'F1_z33':normal_prime0[33]+normal_prime1[33]+normal_prime2[33]+normal_prime3[33], 'F1_z34':normal_prime0[34]+normal_prime1[34]+normal_prime2[34]+normal_prime3[34],'F1_z35':normal_prime0[35]+normal_prime1[35]+normal_prime2[35]+normal_prime3[35],
            'F1_z36':normal_prime0[36]+normal_prime1[36]+normal_prime2[36]+normal_prime3[36], 'F1_z37':normal_prime0[37]+normal_prime1[37]+normal_prime2[37]+normal_prime3[37], 'F1_z38':normal_prime0[38]+normal_prime1[38]+normal_prime2[38]+normal_prime3[38],'F1_z39':normal_prime0[39]+normal_prime1[39]+normal_prime2[39]+normal_prime3[39],
            'F1_z40':normal_prime0[40]+normal_prime1[40]+normal_prime2[40]+normal_prime3[40], 'F1_z41':normal_prime0[41]+normal_prime1[41]+normal_prime2[41]+normal_prime3[41], 'F1_z42':normal_prime0[42]+normal_prime1[42]+normal_prime2[42]+normal_prime3[42],'F1_z43':normal_prime0[43]+normal_prime1[43]+normal_prime2[43]+normal_prime3[43],
            'F1_z44':normal_prime0[44]+normal_prime1[44]+normal_prime2[44]+normal_prime3[44], 'F1_z45':normal_prime0[45]+normal_prime1[45]+normal_prime2[45]+normal_prime3[45], 'F1_z46':normal_prime0[46]+normal_prime1[46]+normal_prime2[46]+normal_prime3[46],'F1_z47':normal_prime0[47]+normal_prime1[47]+normal_prime2[47]+normal_prime3[47],


            'F2_x0': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x1': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x2': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x3': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0],
            'F2_x4': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x5': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x6': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x7': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0],
            'F2_x8': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x9': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x10': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x11': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0],
            'F2_x12': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x13': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x14': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0], 'F2_x15': shear_data_x1[0][0]+shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0],

            'F2_x16':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1], 'F2_x17':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1], 'F2_x18':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1],'F2_x19':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1],
            'F2_x20':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1], 'F2_x21':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1], 'F2_x22':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1],'F2_x23':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1],
            'F2_x24':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1], 'F2_x25':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1], 'F2_x26':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1],'F2_x27':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1],
            'F2_x28':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1], 'F2_x29':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1], 'F2_x30':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1],'F2_x31':shear_data_x1[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1],

            'F2_x32':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2], 'F2_x33':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2], 'F2_x34':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2],'F2_x35':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2],
            'F2_x36':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2], 'F2_x37':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2], 'F2_x38':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2],'F2_x39':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2],
            'F2_x40':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2], 'F2_x41':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2], 'F2_x42':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2],'F2_x43':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2],
            'F2_x44':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2], 'F2_x45':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2], 'F2_x46':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2],'F2_x47':shear_data_x1[0][2]+shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2],

            'F2_y0': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y1': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y2': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y3': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0],
            'F2_y4': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y5': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y6': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y7': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0],
            'F2_y8': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y9': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y10': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y11': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0],
            'F2_y12': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y13': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y14': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0], 'F2_y15': shear_data_y1[1][0]+shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0],

            'F2_y16':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1], 'F2_y17':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1], 'F2_y18':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1],'F2_y19':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1],
            'F2_y20':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1], 'F2_y21':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1], 'F2_y22':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1],'F2_y23':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1],
            'F2_y24':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1], 'F2_y25':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1], 'F2_y26':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1],'F2_y27':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1],
            'F2_y28':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1], 'F2_y29':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1], 'F2_y30':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1],'F2_y31':shear_data_y1[1][1]+shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1],

            'F2_y32':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2], 'F2_y33':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2], 'F2_y34':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2],'F2_y35':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2],
            'F2_y36':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2], 'F2_y37':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2], 'F2_y38':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2],'F2_y39':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2],
            'F2_y40':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2], 'F2_y41':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2], 'F2_y42':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2],'F2_y43':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2],
            'F2_y44':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2], 'F2_y45':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2], 'F2_y46':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2],'F2_y47':shear_data_y1[1][2]+shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2],

            'F2_z0': normal_prime1[0]+normal_prime2[0]+normal_prime3[0]+normal_prime0[0], 'F2_z1': normal_prime1[1]+normal_prime2[1]+normal_prime3[1]+normal_prime0[1], 'F2_z2': normal_prime1[2]+normal_prime2[2]+normal_prime3[2]+normal_prime0[2], 'F2_z3': normal_prime1[3]+normal_prime2[3]+normal_prime3[3]+normal_prime0[3],
            'F2_z4': normal_prime1[4]+normal_prime2[4]+normal_prime3[4]+normal_prime0[4], 'F2_z5': normal_prime1[5]+normal_prime2[5]+normal_prime3[5]+normal_prime0[5], 'F2_z6': normal_prime1[6]+normal_prime2[6]+normal_prime3[6]+normal_prime0[6], 'F2_z7': normal_prime1[7]+normal_prime2[7]+normal_prime3[7]+normal_prime0[7],
            'F2_z8': normal_prime1[8]+normal_prime2[8]+normal_prime3[8]+normal_prime0[8], 'F2_z9': normal_prime1[9]+normal_prime2[9]+normal_prime3[9]+normal_prime0[9], 'F2_z10': normal_prime1[10]+normal_prime2[10]+normal_prime3[10]+normal_prime0[10], 'F2_z11': normal_prime1[11]+normal_prime2[11]+normal_prime3[11]+normal_prime0[11],
            'F2_z12': normal_prime1[12]+normal_prime2[12]+normal_prime3[12]+normal_prime0[12], 'F2_z13': normal_prime1[13]+normal_prime2[13]+normal_prime3[13]+normal_prime0[13], 'F2_z14': normal_prime1[14]+normal_prime2[14]+normal_prime3[14]+normal_prime0[14], 'F2_z15': normal_prime1[15]+normal_prime2[15]+normal_prime3[15]+normal_prime0[15],
            'F2_z16':normal_prime1[16]+normal_prime2[16]+normal_prime3[16]+normal_prime0[16], 'F2_z17':normal_prime1[17]+normal_prime2[17]+normal_prime3[17]+normal_prime0[17], 'F2_z18':normal_prime1[18]+normal_prime2[18]+normal_prime3[18]+normal_prime0[18],'F2_z19':normal_prime1[19]+normal_prime2[19]+normal_prime3[19]+normal_prime0[19],
            'F2_z20':normal_prime1[20]+normal_prime2[20]+normal_prime3[20]+normal_prime0[20], 'F2_z21':normal_prime1[21]+normal_prime2[21]+normal_prime3[21]+normal_prime0[21], 'F2_z22':normal_prime1[22]+normal_prime2[22]+normal_prime3[22]+normal_prime0[22],'F2_z23':normal_prime1[23]+normal_prime2[23]+normal_prime3[23]+normal_prime0[23],
            'F2_z24':normal_prime1[24]+normal_prime2[24]+normal_prime3[24]+normal_prime0[24], 'F2_z25':normal_prime1[25]+normal_prime2[25]+normal_prime3[25]+normal_prime0[25], 'F2_z26':normal_prime1[26]+normal_prime2[26]+normal_prime3[26]+normal_prime0[26],'F2_z27':normal_prime1[27]+normal_prime2[27]+normal_prime3[27]+normal_prime0[27],
            'F2_z28':normal_prime1[28]+normal_prime2[28]+normal_prime3[28]+normal_prime0[28], 'F2_z29':normal_prime1[29]+normal_prime2[29]+normal_prime3[29]+normal_prime0[29], 'F2_z30':normal_prime1[30]+normal_prime2[30]+normal_prime3[30]+normal_prime0[30],'F2_z31':normal_prime1[31]+normal_prime2[31]+normal_prime3[31]+normal_prime0[31],
            'F2_z32':normal_prime1[32]+normal_prime2[32]+normal_prime3[32]+normal_prime0[32], 'F2_z33':normal_prime1[33]+normal_prime2[33]+normal_prime3[33]+normal_prime0[33], 'F2_z34':normal_prime1[34]+normal_prime2[34]+normal_prime3[34]+normal_prime0[34],'F2_z35':normal_prime1[35]+normal_prime2[35]+normal_prime3[35]+normal_prime0[35],
            'F2_z36':normal_prime1[36]+normal_prime2[36]+normal_prime3[36]+normal_prime0[36], 'F2_z37':normal_prime1[37]+normal_prime2[37]+normal_prime3[37]+normal_prime0[37], 'F2_z38':normal_prime1[38]+normal_prime2[38]+normal_prime3[38]+normal_prime0[38],'F2_z39':normal_prime1[39]+normal_prime2[39]+normal_prime3[39]+normal_prime0[39],
            'F2_z40':normal_prime1[40]+normal_prime2[40]+normal_prime3[40]+normal_prime0[40], 'F2_z41':normal_prime1[41]+normal_prime2[41]+normal_prime3[41]+normal_prime0[41], 'F2_z42':normal_prime1[42]+normal_prime2[42]+normal_prime3[42]+normal_prime0[42],'F2_z43':normal_prime1[43]+normal_prime2[43]+normal_prime3[43]+normal_prime0[43],
            'F2_z44':normal_prime1[44]+normal_prime2[44]+normal_prime3[44]+normal_prime0[44], 'F2_z45':normal_prime1[45]+normal_prime2[45]+normal_prime3[45]+normal_prime0[45], 'F2_z46':normal_prime1[46]+normal_prime2[46]+normal_prime3[46]+normal_prime0[46],'F2_z47':normal_prime1[47]+normal_prime2[47]+normal_prime3[47]+normal_prime0[47],

            'F3_x0': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x1': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x2': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x3': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0],
            'F3_x4': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x5': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x6': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x7': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0],
            'F3_x8': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x9': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x10': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x11': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0],
            'F3_x12': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x13': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F_x14': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0], 'F3_x15': shear_data_x2[0][0]+shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0],

            'F3_x16':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1], 'F3_x17':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1], 'F3_x18':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1],'F3_x19':shear_data_x2[0][1]+shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1],
            'F3_x20':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1], 'F3_x21':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1], 'F3_x22':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1],'F3_x23':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1],
            'F3_x24':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1], 'F3_x25':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1], 'F3_x26':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1],'F3_x27':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1],
            'F3_x28':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1], 'F3_x29':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1], 'F3_x30':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1],'F3_x31':shear_data_x2[0][1]+shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1],

            'F3_x32':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2], 'F3_x33':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2], 'F3_x34':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2],'F3_x35':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2],
            'F3_x36':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2], 'F3_x37':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2], 'F3_x38':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2],'F3_x39':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2],
            'F3_x40':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2], 'F3_x41':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2], 'F3_x42':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2],'F3_x43':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2],
            'F3_x44':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2], 'F3_x45':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2], 'F3_x46':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2],'F3_x47':shear_data_x2[0][2]+shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2],

            'F3_y0': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y1': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y2': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y3': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0],
            'F3_y4': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y5': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y6': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y7': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0],
            'F3_y8': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y9': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y10': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y11': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0],
            'F3_y12': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y13': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y14': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0], 'F3_y15': shear_data_y2[1][0]+shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0],

            'F3_y16':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1], 'F3_y17':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1], 'F3_y18':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1],'F3_y19':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1],
            'F3_y20':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1], 'F3_y21':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1], 'F3_y22':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1],'F3_y23':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1],
            'F3_y24':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1], 'F3_y25':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1], 'F3_y26':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1],'F3_y27':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1],
            'F3_y28':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1], 'F3_y29':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1], 'F3_y30':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1],'F3_y31':shear_data_y2[1][1]+shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1],

            'F3_y32':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2], 'F3_y33':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2], 'F3_y34':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2],'F3_y35':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2],
            'F3_y36':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2], 'F3_y37':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2], 'F3_y38':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2],'F3_y39':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2],
            'F3_y40':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2], 'F3_y41':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2], 'F3_y42':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2],'F3_y43':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2],
            'F3_y44':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2], 'F3_y45':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2], 'F3_y46':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2],'F3_y47':shear_data_y2[1][2]+shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2],

            'F3_z0': normal_prime2[0]+normal_prime3[0]+normal_prime0[0]+normal_prime1[0], 'F3_z1': normal_prime2[1]+normal_prime3[1]+normal_prime0[1]+normal_prime1[1], 'F3_z2': normal_prime2[2]+normal_prime3[2]+normal_prime0[2]+normal_prime1[2], 'F3_z3': normal_prime2[3]+normal_prime3[3]+normal_prime0[3]+normal_prime1[3],
            'F3_z4': normal_prime2[4]+normal_prime3[4]+normal_prime0[4]+normal_prime1[4], 'F3_z5': normal_prime2[5]+normal_prime3[5]+normal_prime0[5]+normal_prime1[5], 'F3_z6': normal_prime2[6]+normal_prime3[6]+normal_prime0[6]+normal_prime1[6], 'F3_z7': normal_prime2[7]+normal_prime3[7]+normal_prime0[7]+normal_prime1[7],
            'F3_z8': normal_prime2[8]+normal_prime3[8]+normal_prime0[8]+normal_prime1[8], 'F3_z9': normal_prime2[9]+normal_prime3[9]+normal_prime0[9]+normal_prime1[9], 'F3_z10': normal_prime2[10]+normal_prime3[10]+normal_prime0[10]+normal_prime1[10], 'F3_z11': normal_prime2[11]+normal_prime3[11]+normal_prime0[11]+normal_prime1[11],
            'F3_z12': normal_prime2[12]+normal_prime3[12]+normal_prime0[12]+normal_prime1[12], 'F3_z13': normal_prime2[13]+normal_prime3[13]+normal_prime0[13]+normal_prime1[13], 'F3_z14': normal_prime2[14]+normal_prime3[14]+normal_prime0[14]+normal_prime1[14], 'F3_z15': normal_prime2[15]+normal_prime3[15]+normal_prime0[15]+normal_prime1[15],
            'F3_z16':normal_prime2[16]+normal_prime3[16]+normal_prime0[16]+normal_prime1[16], 'F3_z17':normal_prime2[17]+normal_prime3[17]+normal_prime0[17]+normal_prime1[17], 'F3_z18':normal_prime2[18]+normal_prime3[18]+normal_prime0[18]+normal_prime1[18],'F3_z19':normal_prime2[19]+normal_prime3[19]+normal_prime0[19]+normal_prime1[19],
            'F3_z20':normal_prime2[20]+normal_prime3[20]+normal_prime0[20]+normal_prime1[20], 'F3_z21':normal_prime2[21]+normal_prime3[21]+normal_prime0[21]+normal_prime1[21], 'F3_z22':normal_prime2[22]+normal_prime3[22]+normal_prime0[22]+normal_prime1[22],'F3_z23':normal_prime2[23]+normal_prime3[23]+normal_prime0[23]+normal_prime1[23],
            'F3_z24':normal_prime2[24]+normal_prime3[24]+normal_prime0[24]+normal_prime1[24], 'F3_z25':normal_prime2[25]+normal_prime3[25]+normal_prime0[25]+normal_prime1[25], 'F3_z26':normal_prime2[26]+normal_prime3[26]+normal_prime0[26]+normal_prime1[26],'F3_z27':normal_prime2[27]+normal_prime3[27]+normal_prime0[27]+normal_prime1[27],
            'F3_z28':normal_prime2[28]+normal_prime3[28]+normal_prime0[28]+normal_prime1[28], 'F3_z29':normal_prime2[29]+normal_prime3[29]+normal_prime0[29]+normal_prime1[29], 'F3_z30':normal_prime2[30]+normal_prime3[30]+normal_prime0[30]+normal_prime1[30],'F3_z31':normal_prime2[31]+normal_prime3[31]+normal_prime0[31]+normal_prime1[31],
            'F3_z32':normal_prime2[32]+normal_prime3[32]+normal_prime0[32]+normal_prime1[32], 'F3_z33':normal_prime2[33]+normal_prime3[33]+normal_prime0[33]+normal_prime1[33], 'F3_z34':normal_prime2[34]+normal_prime3[34]+normal_prime0[34]+normal_prime1[34],'F3_z35':normal_prime2[35]+normal_prime3[35]+normal_prime0[35]+normal_prime1[35],
            'F3_z36':normal_prime2[36]+normal_prime3[36]+normal_prime0[36]+normal_prime1[36], 'F3_z37':normal_prime2[37]+normal_prime3[37]+normal_prime0[37]+normal_prime1[37], 'F3_z38':normal_prime2[38]+normal_prime3[38]+normal_prime0[38]+normal_prime1[38],'F3_z39':normal_prime2[39]+normal_prime3[39]+normal_prime0[39]+normal_prime1[39],
            'F3_z40':normal_prime2[40]+normal_prime3[40]+normal_prime0[40]+normal_prime1[40], 'F3_z41':normal_prime2[41]+normal_prime3[41]+normal_prime0[41]+normal_prime1[41], 'F3_z42':normal_prime2[42]+normal_prime3[42]+normal_prime0[42]+normal_prime1[42],'F3_z43':normal_prime2[43]+normal_prime3[43]+normal_prime0[43]+normal_prime1[43],
            'F3_z44':normal_prime2[44]+normal_prime3[44]+normal_prime0[44]+normal_prime1[44], 'F3_z45':normal_prime2[45]+normal_prime3[45]+normal_prime0[45]+normal_prime1[45], 'F3_z46':normal_prime2[46]+normal_prime3[46]+normal_prime0[46]+normal_prime1[46],'F3_z47':normal_prime2[47]+normal_prime3[47]+normal_prime0[47]+normal_prime1[47],


            'F4_x0': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x1': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x2': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x3': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0],
            'F4_x4': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x5': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x6': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x7': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0],
            'F4_x8': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x9': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x10': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x11': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0],
            'F4_x12': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x13': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x14': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0], 'F4_x15': shear_data_x3[0][0]+shear_data_x0[0][0]+shear_data_x1[0][0]+shear_data_x2[0][0],

            'F4_x16':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1], 'F4_x17':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1], 'F4_x18':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1],'F4_x19':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1],
            'F4_x20':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1], 'F4_x21':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1], 'F4_x22':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1],'F4_x23':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1],
            'F4_x24':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1], 'F4_x25':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1], 'F4_x26':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1],'F4_x27':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1],
            'F4_x28':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1], 'F4_x29':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1], 'F4_x30':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1],'F4_x31':shear_data_x3[0][1]+shear_data_x0[0][1]+shear_data_x1[0][1]+shear_data_x2[0][1],

            'F4_x32':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2], 'F4_x33':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2], 'F4_x34':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2],'F4_x35':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2],
            'F4_x36':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2], 'F4_x37':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2], 'F4_x38':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2],'F4_x39':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2],
            'F4_x40':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2], 'F4_x41':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2], 'F4_x42':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2],'F4_x43':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2],
            'F4_x44':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2], 'F4_x45':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2], 'F4_x46':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2],'F4_x47':shear_data_x3[0][2]+shear_data_x0[0][2]+shear_data_x1[0][2]+shear_data_x2[0][2],

            'F4_y0': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y1': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y2': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y3': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0],
            'F4_y4': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y5': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y6': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y7': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0],
            'F4_y8': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y9': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y10': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y11': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0],
            'F4_y12': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y13': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y14': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0], 'F4_y15': shear_data_y3[1][0]+shear_data_y0[1][0]+shear_data_y1[1][0]+shear_data_y2[1][0],

            'F4_y16':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1], 'F4_y17':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1], 'F4_y18':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1],'F4_y19':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1],
            'F4_y20':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1], 'F4_y21':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1], 'F4_y22':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1],'F4_y23':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1],
            'F4_y24':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1], 'F4_y25':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1], 'F4_y26':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1],'F4_y27':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1],
            'F4_y28':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1], 'F4_y29':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1], 'F4_y30':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1],'F4_y31':shear_data_y3[1][1]+shear_data_y0[1][1]+shear_data_y1[1][1]+shear_data_y2[1][1],

            'F4_y32':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2], 'F4_y33':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2], 'F4_y34':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2],'F4_y35':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2],
            'F4_y36':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2], 'F4_y37':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2], 'F4_y38':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2],'F4_y39':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2],
            'F4_y40':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2], 'F4_y41':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2], 'F4_y42':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2],'F4_y43':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2],
            'F4_y44':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2], 'F4_y45':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2], 'F4_y46':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2],'F4_y47':shear_data_y3[1][2]+shear_data_y0[1][2]+shear_data_y1[1][2]+shear_data_y2[1][2],

            'F4_z0': normal_prime3[0]+normal_prime0[0]+normal_prime1[0]+normal_prime2[0], 'F4_z1': normal_prime3[1]+normal_prime0[1]+normal_prime1[1]+normal_prime2[1], 'F4_z2': normal_prime3[2]+normal_prime0[0]+normal_prime1[0]+normal_prime2[0], 'F4_z3': normal_prime3[3]+normal_prime0[3]+normal_prime1[3]+normal_prime2[3],
            'F4_z4': normal_prime3[4]+normal_prime0[4]+normal_prime1[4]+normal_prime2[4], 'F4_z5': normal_prime3[5]+normal_prime0[5]+normal_prime1[5]+normal_prime2[5], 'F4_z6': normal_prime3[6]+normal_prime0[6]+normal_prime1[6]+normal_prime2[6], 'F4_z7': normal_prime3[7]+normal_prime0[7]+normal_prime1[7]+normal_prime2[7],
            'F4_z8': normal_prime3[8]+normal_prime0[8]+normal_prime1[8]+normal_prime2[8], 'F4_z9': normal_prime3[9]+normal_prime0[9]+normal_prime1[9]+normal_prime2[9], 'F4_z10': normal_prime3[10]+normal_prime0[10]+normal_prime1[10]+normal_prime2[10], 'F4_z11': normal_prime3[11]+normal_prime0[11]+normal_prime1[11]+normal_prime2[11],
            'F4_z12':normal_prime3[12]+normal_prime0[12]+normal_prime1[12]+normal_prime2[12], 'F4_z13': normal_prime3[13]+normal_prime0[13]+normal_prime1[13]+normal_prime2[13], 'F4_z14': normal_prime3[14]+normal_prime0[14]+normal_prime1[14]+normal_prime2[14], 'F4_z15': normal_prime3[15]+normal_prime0[15]+normal_prime1[15]+normal_prime2[15],
            'F4_z16':normal_prime3[16]+normal_prime0[16]+normal_prime1[16]+normal_prime2[16], 'F4_z17':normal_prime3[17]+normal_prime0[17]+normal_prime1[17]+normal_prime2[17], 'F4_z18':normal_prime3[18]+normal_prime0[18]+normal_prime1[18]+normal_prime2[18],'F4_z19':normal_prime3[19]+normal_prime0[19]+normal_prime1[19]+normal_prime2[19],
            'F4_z20':normal_prime3[20]+normal_prime0[20]+normal_prime1[20]+normal_prime2[20], 'F4_z21':normal_prime3[21]+normal_prime0[21]+normal_prime1[21]+normal_prime2[21], 'F4_z22':normal_prime3[22]+normal_prime0[22]+normal_prime1[22]+normal_prime2[22],'F4_z23':normal_prime3[23]+normal_prime0[23]+normal_prime1[23]+normal_prime2[23],
            'F4_z24':normal_prime3[24]+normal_prime0[24]+normal_prime1[24]+normal_prime2[24], 'F4_z25':normal_prime3[25]+normal_prime0[25]+normal_prime1[25]+normal_prime2[25], 'F4_z26':normal_prime3[26]+normal_prime0[26]+normal_prime1[26]+normal_prime2[26],'F4_z27':normal_prime3[27]+normal_prime0[27]+normal_prime1[27]+normal_prime2[27],
            'F4_z28':normal_prime3[28]+normal_prime0[28]+normal_prime1[28]+normal_prime2[28], 'F4_z29':normal_prime3[29]+normal_prime0[29]+normal_prime1[29]+normal_prime2[29], 'F4_z30':normal_prime3[30]+normal_prime0[30]+normal_prime1[30]+normal_prime2[30],'F4_z31':normal_prime3[31]+normal_prime0[31]+normal_prime1[31]+normal_prime2[31],
            'F4_z32':normal_prime3[32]+normal_prime0[32]+normal_prime1[32]+normal_prime2[32], 'F4_z33':normal_prime3[33]+normal_prime0[33]+normal_prime1[33]+normal_prime2[33], 'F4_z34':normal_prime3[34]+normal_prime0[34]+normal_prime1[34]+normal_prime2[34],'F4_z35':normal_prime3[35]+normal_prime0[35]+normal_prime1[35]+normal_prime2[35],
            'F4_z36':normal_prime3[36]+normal_prime0[36]+normal_prime1[36]+normal_prime2[36], 'F4_z37':normal_prime3[37]+normal_prime0[37]+normal_prime1[37]+normal_prime2[37], 'F4_z38':normal_prime3[38]+normal_prime0[38]+normal_prime1[38]+normal_prime2[38],'F4_z39':normal_prime3[39]+normal_prime0[39]+normal_prime1[39]+normal_prime2[39],
            'F4_z40':normal_prime3[40]+normal_prime0[40]+normal_prime1[40]+normal_prime2[40], 'F4_z41':normal_prime3[41]+normal_prime0[41]+normal_prime1[41]+normal_prime2[41], 'F4_z42':normal_prime3[42]+normal_prime0[42]+normal_prime1[42]+normal_prime2[42],'F4_z43':normal_prime3[43]+normal_prime0[43]+normal_prime1[43]+normal_prime2[43],
            'F4_z44':normal_prime3[44]+normal_prime0[44]+normal_prime1[44]+normal_prime2[44], 'F4_z45':normal_prime3[45]+normal_prime0[45]+normal_prime1[45]+normal_prime2[45], 'F4_z46':normal_prime3[46]+normal_prime0[46]+normal_prime1[46]+normal_prime2[46],'F4_z47':normal_prime3[47]+normal_prime0[47]+normal_prime1[47]+normal_prime2[47],

            'label':label})

            #Raw_prime_Filename="Raw_Prime_data_Class_"+str(label[0])+"_oriLen_"+str(len(normal_prime3[45]))+"_num_"+str(file_num)+".csv"

            #create csv file
            frame=frame.sample(frac=1,random_state=1)
            #determine length
            #frame=frame.iloc[:,1:]# remove index column
            Raw_prime_Filename = "Raw_Prime_data_Class_" + str(label[0]) + "_oriLen_" + str(
                len(frame)) + "_num_" + str(file_num) + ".csv"
            frame.to_csv(Raw_prime_Folder_path+Raw_prime_Filename,index=False,sep=',')
            print("saved raw_prime_csv file:"+str(file_num))

def collate_the_csv(Raw_prime_Folder_path,out_path):
    file_num = 0
    for filename in os.listdir(Raw_prime_Folder_path):
        if os.path.splitext(filename)[-1] == '.csv':
            if file_num ==0:
                first_file=pd.read_csv(os.path.join(Raw_prime_Folder_path,filename))

                #first_file = first_file.iloc[:, 1:]# remove index column
                first_file.to_csv(out_path+'Dataset_NUSG.csv', index=False, sep=',',header=True)
            else:
                rest_file = pd.read_csv(os.path.join(Raw_prime_Folder_path, filename))
                rest_file.to_csv(out_path+'Dataset_NUSG.csv', index=False, sep=',',header=False,mode='a+')
            file_num += 1
    print('prpcessed {0} files'.format(file_num,int))

def shuffle_csv(out_file,TRAINRATIO=0.8, VALRATIO=0.1):

    #random.shuffle(out_file)#dataset =dataset.shuffle()
    data=pd.read_csv(out_file)
    data=shuffle(data)
    #data_header = pd.read_csv(out_file,nrows=0)
    #data_header.to_csv('./biotacsp/raw/shuffled.csv',index=False)

    data.to_csv('./biotacsp/raw/shuffled.csv',index=False,index_label=None)

    #split the data
    train_dataset=pd.read_csv('./biotacsp/raw/shuffled.csv')
    train_dataset=train_dataset.iloc[:int(TRAINRATIO * len(train_dataset))]
    #data_header = pd.read_csv(out_file, nrows=0)
    #data_header.to_csv('./biotacsp/raw/train_dataset.csv',index=False,sep=',')
    train_dataset.to_csv('./biotacsp/raw/train_dataset.csv',index=False,sep=',',index_label=None)

    val_dataset = pd.read_csv('./biotacsp/raw/shuffled.csv')
    val_dataset = val_dataset.iloc[int(TRAINRATIO * len(val_dataset)):int((TRAINRATIO + VALRATIO) * len(val_dataset))]
    #data_header = pd.read_csv(out_file, nrows=0)
    #data_header.to_csv('./biotacsp/raw/val_dataset.csv',index=False,sep=',')
    val_dataset.to_csv('./biotacsp/raw/val_dataset.csv', index=False,sep=',',index_label=None)

    test_dataset = pd.read_csv('./biotacsp/raw/shuffled.csv')
    test_dataset = test_dataset.iloc[int((TRAINRATIO + VALRATIO) * len(test_dataset)):]
    #data_header = pd.read_csv(out_file, nrows=0)
    #data_header.to_csv('./biotacsp/raw/test_dataset.csv',index=False,sep=',')
    test_dataset.to_csv('./biotacsp/raw/test_dataset.csv', index=False,sep=',',index_label=None)

    #finger1_x_data = test_dataset = pd.read_csv('./biotacsp/raw/shuffled.csv')#np.copy(sample_.iloc[0:48]).astype(np.int, copy=False)
    #finger1_x_data=finger1_x_data.iloc[2]
    #finger1_x_data=finger1_x_data.iloc[3]
    #print(finger1_x_data)




    #train_dataset = dataset[:int(TRAINRATIO * len(dataset))]
    #val_dataset = dataset[int(TRAINRATIO * len(dataset)):int((TRAINRATIO + VALRATIO) * len(dataset))]
    #test_dataset = dataset[int((TRAINRATIO + VALRATIO) * len(dataset)):]
    #return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    #"clear C:\Users\Lenovo\Desktop\tactile-gcn-12features\data\biotacsp\NUSG  before run"
    flag_create_csv=True
    flag_shuffle_csv=True
    if flag_create_csv:
        Raw_prime_Folder_path='C:\\Users\\Lenovo\\Desktop\\tactile-gcn-12features\\data\\biotacsp\\NUSG\\'
        data_dir='C:\\Users\\Lenovo\\Desktop\\tactile-gcn-12features\\data\\biotacsp\\raw_sampled\\'
        out_path='C:\\Users\\Lenovo\\Desktop\\tactile-gcn-12features\\data\\biotacsp\\raw\\'

        create_csv_raws(data_dir, Raw_prime_Folder_path)
        collate_the_csv(Raw_prime_Folder_path,out_path)
    if flag_shuffle_csv:
        out_file = './biotacsp/raw/Dataset_NUSG.csv'
        shuffle_csv(out_file)
    print('done')

