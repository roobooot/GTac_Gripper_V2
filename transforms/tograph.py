import logging

import torch
from torch_geometric.data import Data
import numpy as np
import scipy.spatial

log = logging.getLogger(__name__)

#generate the full connect edges
#input:NA
#output: start and end of the edgdes
def generate_edge_mx():
    start_pos_mx=[0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    end_pos_mx=[1, 4, 5, 0, 2, 4, 5, 6, 1, 3, 5, 6, 7, 2, 6, 7]
    for i in range(4,44):
        if (i%4)==0:
            for j in range(5):
                start_pos_mx.append(i)
            end_pos_mx.append(i - 4)
            end_pos_mx.append(i - 3)
            end_pos_mx.append(i + 1)
            end_pos_mx.append(i + 4)
            end_pos_mx.append(i + 5)
        if (i-1)%4 ==0 or (i-2)%4==0:
            for j in range(8):
                start_pos_mx.append(i)
            end_pos_mx.append(i - 5)
            end_pos_mx.append(i - 4)
            end_pos_mx.append(i - 3)
            end_pos_mx.append(i - 1)
            end_pos_mx.append(i + 1)
            end_pos_mx.append(i + 3)
            end_pos_mx.append(i + 4)
            end_pos_mx.append(i + 5)
        if (i+1) % 4 == 0:
            for j in range(5):
                start_pos_mx.append(i)
            end_pos_mx.append(i - 5)
            end_pos_mx.append(i - 4)
            end_pos_mx.append(i - 1)
            end_pos_mx.append(i + 3)
            end_pos_mx.append(i + 4)
    start_pos_mx.append(44);start_pos_mx.append(44);start_pos_mx.append(44);start_pos_mx.append(45);start_pos_mx.append(45);start_pos_mx.append(45);
    start_pos_mx.append(45);start_pos_mx.append(45);start_pos_mx.append(46);start_pos_mx.append(46);start_pos_mx.append(46);start_pos_mx.append(46);
    start_pos_mx.append(46);start_pos_mx.append(47);start_pos_mx.append(47);start_pos_mx.append(47)
    end_pos_mx.append(40);end_pos_mx.append(41);end_pos_mx.append(45);end_pos_mx.append(40);end_pos_mx.append(41);end_pos_mx.append(42);
    end_pos_mx.append(44);end_pos_mx.append(46);end_pos_mx.append(41);end_pos_mx.append(42);end_pos_mx.append(43);end_pos_mx.append(45);
    end_pos_mx.append(47);end_pos_mx.append(42);end_pos_mx.append(43);end_pos_mx.append(46)
    #print(start_pos_mx)
    #print(end_pos_mx)
    return start_pos_mx,end_pos_mx

class ToGraph(object):

    def __init__(self, k):

        assert(k >= 0), 'graph_k must be equal or greater than 0'

        # Actually, this would be X
        self.m_taxels_y = [0.386434851,0.318945051,0.08737268,0.083895199,-0.018624877,-0.091886816,-0.1366595,-0.223451775,-0.320752549,-0.396931929,0.386434851,0.318945051,0.08737268,0.083895199,-0.018624877,-0.091886816,-0.1366595,-0.223451775,-0.320752549,-0.396931929,0.25875305,0.170153841,0.170153841,0.075325086,-0.108966104,-0.205042252,-0.128562247,-0.235924865,-0.30011705,-0.12043608,-0.237549685,-0.270674659,-0.199498368,-0.100043884,-0.108966104,-0.205042252,-0.128562247,-0.235924865,-0.30011705,-0.12043608,-0.237549685,-0.270674659,-0.199498368,-0.100043884,-0.252337663,-0.274427927,-0.274427927,-0.187122746]
        # Actually, this would be Y
        self.m_taxels_z = [-0.108966104,-0.205042252,-0.128562247,-0.235924865,-0.30011705,-0.12043608,-0.237549685,-0.270674659,-0.199498368,-0.100043884,-0.108966104,-0.205042252,-0.128562247,-0.235924865,-0.30011705,-0.12043608,-0.237549685,-0.270674659,-0.199498368,-0.100043884,-0.252337663,-0.274427927,-0.274427927,-0.298071391,0.156871012,0.12070609,0.281981384,0.201566857,0.094918748,0.284956139,0.187122746,0.071536904,0.127771244,0.151565706,-0.156871012,-0.12070609,-0.281981384,-0.201566857,-0.094918748,-0.284956139,-0.187122746,-0.071536904,-0.127771244,-0.151565706,0,0.072909607,-0.072909607,0]
        # Actually, this would be Z
        self.m_taxels_x = [0.156871012,0.12070609,0.281981384,0.201566857,0.094918748,0.284956139,0.187122746,0.071536904,0.127771244,0.151565706,-0.156871012,-0.12070609,-0.281981384,-0.201566857,-0.094918748,-0.284956139,-0.187122746,-0.071536904,-0.127771244,-0.151565706,0,0.072909607,-0.072909607,0,0.386434851,0.318945051,0.08737268,0.083895199,-0.018624877,-0.091886816,-0.1366595,-0.223451775,-0.320752549,-0.396931929,0.386434851,0.318945051,0.08737268,0.083895199,-0.018624877,-0.091886816,-0.1366595,-0.223451775,-0.320752549,-0.396931929,0.25875305,0.170153841,0.170153841,0.075325086]

        if k == 0: ## Use manual connections
            '''
            self.m_edge_origins =   [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3,
						 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
						 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
						 12, 12, 12, 12, 12,
                         13, 13, 13, 13, 13, 13, 13, 13,
                         14, 14, 14, 14, 14, 15, 15, 15]
            self.m_edge_ends =      [1, 4, 5, 0, 2, 4, 5, 6, 1, 3, 5, 6, 7, 2, 6, 7,
					 0, 1, 5, 8, 9, 0, 1, 2, 4, 6, 8, 9, 10, 1, 2, 3, 5, 7, 9, 10, 11, 2, 3, 6, 10, 11,
					 4, 5, 9, 12, 13, 4, 5, 6, 8, 10, 12, 13, 14, 5, 6, 7, 9, 11, 13, 14, 15, 6, 7, 10, 14, 15,
					 8, 9, 13, 16, 17,
                     8, 9, 10, 12, 14, 16, 17, 18,
                     9, 10, 11, 13, 15, 10, 11, 14]
            '''
            self.m_edge_origins,self.m_edge_ends= generate_edge_mx()

        else:
            points_ = np.transpose(np.vstack((self.m_taxels_x, self.m_taxels_y)), (1, 0))
            tree_ = scipy.spatial.KDTree(points_)

            _, idxs_ = tree_.query(points_, k=k + 1) # Closest point will be the point itself, so k + 1
            idxs_ = idxs_[:, 1:] # Remove closest point, which is the point itself
        
            self.m_edge_origins = np.repeat(np.arange(len(points_)), k)
            self.m_edge_ends = np.reshape(idxs_, (-1))

    def __call__(self, sample):

        # Index finger
        graph_x_ = torch.tensor(np.vstack((sample['F1_x'], sample['F1_y'], sample['F1_z'],
                                           sample['F2_x'], sample['F2_y'], sample['F2_z'],
                                           sample['F3_x'], sample['F3_y'], sample['F3_z'],
                                           sample['F4_x'], sample['F4_y'], sample['F4_z'])), dtype=torch.float).transpose(0, 1)
        graph_edge_index_ = torch.tensor([self.m_edge_origins, self.m_edge_ends], dtype=torch.long)
        graph_pos_ = torch.tensor(np.vstack((self.m_taxels_x, self.m_taxels_y, self.m_taxels_z)), dtype=torch.float).transpose(0, 1)
        graph_y_ = torch.tensor([sample['object']], dtype=torch.long)

        data_ = Data(x = graph_x_,
                    edge_index = graph_edge_index_,
                    pos = graph_pos_,
                    y = graph_y_)

        return data_

    def __repr__(self):
        return "{}".format(self.__class__.__name__)

#generate_edge_mx()