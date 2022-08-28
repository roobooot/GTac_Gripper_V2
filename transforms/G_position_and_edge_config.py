import matplotlib.pyplot as plt
#import networkx as nx
import numpy as np

import torch

from torch_geometric.data import Data

#import GTac_Data
#import gtac_config


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

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————
# For GCN graph modeling of single sensor chip
def create_edge_single(index_matrix):
		start = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3,
						 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
						 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
						 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15]
		end = [1, 4, 5, 0, 2, 4, 5, 6, 1, 3, 5, 6, 7, 2, 6, 7,
					 0, 1, 5, 8, 9, 0, 1, 2, 4, 6, 8, 9, 10, 1, 2, 3, 5, 7, 9, 10, 11, 2, 3, 6, 10, 11,
					 4, 5, 9, 12, 13, 4, 5, 6, 8, 10, 12, 13, 14, 5, 6, 7, 9, 11, 13, 14, 15, 6, 7, 10, 14, 15,
					 8, 9, 13, 8, 9, 10, 12, 14, 9, 10, 11, 13, 15, 10, 11, 14]
		start = index_matrix[start]
		end = index_matrix[end]
		return start, end


def connect_between_phalanx(si0, si1):
		# print(f'si0 is {si0} and si1 is {si1}')
		sec_index = np.append(si0[:4], si1[12:16])
		# print(f'connection: sec_index is {sec_index}')
		start = [4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7,
						 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
		end = [0, 1, 5, 0, 1, 2, 4, 6, 1, 2, 3, 5, 7, 2, 3, 6,
					 1, 4, 5, 0, 2, 4, 5, 6, 1, 3, 5, 6, 7, 2, 6, 7]
		start = sec_index[start]
		end = sec_index[end]
		return start, end


def connect_within_palm():
		sec_index = []
		for f in range(5):
				if f != 2:
						# print(f'finger {f}, sec_index is {gtac_data.gtac_data.find_sec_index(f, 0)}')
						if f > 2:
								f = f-1
						array = np.array([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]) + 48 * f
						sec_index.append(array)
		sec_index = [sec_index[i][12:16] for i in range(4)]
		sec_index = np.reshape(sec_index, 16)
		start = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]
		end = [15, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 0]
		start = sec_index[start]
		end = sec_index[end]
		# print(f'start is {start}, end is {end}')
		return start, end


def create_edge_index_single_p2t(f, s):
		# create node that starts from the palm to fingertip
		# original matrix start at the bottom left corner, [[12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
		# then based on the mapping between indexing and edge_matrix, generate the edge information by correct index
		sec_index = GTac_Data.gtac_data.find_sec_index(f, s)  # get a 1*19 array
		# print(f'sec_index is {sec_index}')
		sec_index = np.reshape(sec_index[:16], 16)
		single_edge_index = create_edge_single(sec_index)
		return single_edge_index


def create_edge_index_finger_p2t(f):
		# original matrix start at the bottom left corner, [[12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
		# then based on the mapping between indexing and edge_matrix, generate the edge information by correct index
		if f > 2:
				f -= 1
		array = np.array([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]) + 48*f
		sec_index_0 = array
		# sec_index_0 = np.reshape(sec_index_0[:16], (4, 4))
		sec_index_1 = array+16
		sec_index_2 = array+32
		start0, end0 = create_edge_single(sec_index_0)
		start1, end1 = create_edge_single(sec_index_1)
		start2, end2 = create_edge_single(sec_index_2)
		# print(f'start0 is {start0}')
		# print(f'start1 is {start1}')
		# print(f'start2 is {start2}')
		cs1, ce1 = connect_between_phalanx(sec_index_0, sec_index_1)
		cs2, ce2 = connect_between_phalanx(sec_index_1, sec_index_2)
		# print(f'sec_index1 is {sec_index_1} and sec_index2 is {sec_index_2}')
		# print(f'cs2 is {cs2} and ce2 is {ce2}')
		start = np.concatenate((start0, start1, start2, cs1, cs2), axis=0)
		# print(f'start is {start}')
		end = np.concatenate((end0,  end1, end2, ce1, ce2), axis=0)
		return np.array([start, end])


def create_edge_index_finger_p2t_2(f):
		# original matrix start at the bottom left corner, [[12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
		# then based on the mapping between indexing and edge_matrix, generate the edge information by correct index
		sec_index_0 = np.array(GTac_Data.gtac_data.find_sec_index(f, 0))
		# sec_index_0 = np.reshape(sec_index_0[:16], (4, 4))
		sec_index_1 = np.array(GTac_Data.gtac_data.find_sec_index(f, 1))
		sec_index_2 = np.array(GTac_Data.gtac_data.find_sec_index(f, 2))
		start0, end0 = create_edge_single(sec_index_0)
		start1, end1 = create_edge_single(sec_index_1)
		start2, end2 = create_edge_single(sec_index_2)
		# print(f'start0 is {start0}')
		# print(f'start1 is {start1}')
		# print(f'start2 is {start2}')
		cs1, ce1 = connect_between_phalanx(sec_index_0, sec_index_1)
		cs2, ce2 = connect_between_phalanx(sec_index_1, sec_index_2)
		# print(f'sec_index1 is {sec_index_1} and sec_index2 is {sec_index_2}')
		# print(f'cs2 is {cs2} and ce2 is {ce2}')
		start = np.concatenate((start0, start1, start2, cs1, cs2), axis=0)
		# print(f'start is {start}')
		end = np.concatenate((end0, end1, end2, ce1, ce2), axis=0)
		return np.array([start, end])


def create_edge_index_whole_p2t():
		edge_index_f0 = create_edge_index_finger_p2t(f=0)
		edge_index_f1 = create_edge_index_finger_p2t(f=1)
		edge_index_f3 = create_edge_index_finger_p2t(f=3)
		edge_index_f4 = create_edge_index_finger_p2t(f=4)
		palm_start, palm_end = connect_within_palm()
		start = np.concatenate((edge_index_f0[0], edge_index_f1[0], edge_index_f3[0], edge_index_f4[0], palm_start), axis=0)
		end = np.concatenate((edge_index_f0[1], edge_index_f1[1], edge_index_f3[1], edge_index_f4[1], palm_end), axis=0)
		return np.array([start, end])


def create_edge_index_palm():
		edge_index_f0 = create_edge_index_single_p2t(f=0, s=0)
		edge_index_f1 = create_edge_index_single_p2t(f=1, s=0)
		edge_index_f3 = create_edge_index_single_p2t(f=2, s=0)
		edge_index_f4 = create_edge_index_single_p2t(f=3, s=0)
		palm_start, palm_end = connect_within_palm()
		start = np.concatenate((edge_index_f0[0], edge_index_f1[0], edge_index_f3[0], edge_index_f4[0], palm_start), axis=0)
		end = np.concatenate((edge_index_f0[1], edge_index_f1[1], edge_index_f3[1], edge_index_f4[1], palm_end), axis=0)
		return np.array([start, end])



def feature_single_section(dataframe, f=0, s=2):
		# reshape the dataset for train in GCN, for test on single section
		# input shape: [1*285]
		# output shape: [N, 15, 19, 1]
		# in each sec (19 signals): [mat1, mat2, ..., mat16, magx, magy, magz]
		# Define the node feature X
		X = []
		Pos = []
		saii_index = f * 9 + (2 - s) * 3
		# sum_value, press_location_r, press_location_c = gtac_data.gtac_data.find_FAI_sum_press_loc(dataframe, f, s)
		x_train_saii_sec = dataframe[saii_index:saii_index + 3]
		saii = [x / gtac_config.MAT_NUM for x in x_train_saii_sec]
		for i in range(gtac_config.MAT_NUM):  # MAT_NUM -> 16: there 4*4 sensing points on FA-I layer
				r = i // 4
				c = i % 4
				fai_index = GTac_Data.gtac_data.find_FAI_index(f, s, r, c)
				# print(f'fai_index is {fai_index}')
				fai_r_c = dataframe[fai_index]
				fai = [fai_r_c, 0, 0]
				a = list(np.add(saii, fai))
				X.append(a)
				Pos.append([r-1.5, c-1.5, 0])
		# print(f'Features are {X}')
		return X, Pos


def feature_single_finger(dataframe, f=0):
		X = []
		Pos = []
		for i in range(3):
				x, pos = feature_single_section(dataframe, f, i)
				X = X + x
				Pos = Pos + pos
		return X, Pos


def feature_extration(dataframe):
		X = []
		Pos = []
		for f in range(5):
				if f != 2:
						x, pos = feature_single_finger(dataframe, f)
						X = X + x
						Pos = Pos + pos
		return X, Pos


def create_edge_index_single(f, s):
		# create node that starts from the fingertip to palm
		# original matrix start at the bottom left corner, [[12, 13, 14, 15], [8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
		# transform it into [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
		# then based on the mapping between indexing and edge_matrix, generate the edge information by correct index
		sec_index = GTac_Data.gtac_data.find_sec_index(f, s)
		# print(f'sec_index before processing is {sec_index}')
		sec_index = np.reshape(sec_index[:16], (4, 4))[::-1][:]
		sec_index = np.reshape(sec_index, 16)
		# print(f'sec_index after processing is {sec_index}')
		single_edge_index = create_edge_single(sec_index)
		return single_edge_index


# construct_graph_and_visualize()
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————
# implementation of Graph representation
def construct_graph_finger(dataframe, f):
		# for single section, example: f=0
		# 1. node features extraction
		X, pos = feature_single_finger(dataframe, f=f)
		X = torch.tensor(X, dtype=torch.float)
		pos = torch.tensor(pos, dtype=torch.float)

		# 2. contruct the edge_index
		edge_index = create_edge_index_finger_p2t(f=f)
		edge_index = torch.tensor(edge_index, dtype=torch.long)

		# 3. Construct of the data class
		data = Data(x=X, edge_index=edge_index, pos=pos)

		print(f'data.keys {data.keys}')
		print(f'num_node is {data.num_nodes}')
		print(f'num_node_features is {data.num_node_features}')
		print(f'data.pos is {data.pos}')
		print(f'num_features is {data.num_features}')
		print(f'num_edges is {data.num_edges}')
		print(f'edge_index is {data.edge_index}')
		print(f'num_edge_features is {data.num_edge_features}')
		# visualize_graph(data, 'g')


def construct_graph(dataframe, label):
		# for single section, example: f=0
		# 1. node features extraction
		X, pos = feature_extration(dataframe)
		X = torch.tensor(X, dtype=torch.float)
		pos = torch.tensor(pos, dtype=torch.float)
		print(f'in construct_graph, X: {X.shape} ')

		# 2. contruct the edge_index
		edge_index = create_edge_index_whole_p2t()
		edge_index = torch.tensor(edge_index, dtype=torch.long)
		print(f'in construct_graph, edge_index: {edge_index.shape} ')
		print(f'in construct_graph, edge_index max is {edge_index.min(), edge_index.max()} ')

		# 3. Construct of the data class
		data = Data(x=X, edge_index=edge_index, pos=pos, y=label)

		# print(f'data.keys {data.keys}')
		# print(f'num_node is {data.num_nodes}')
		# print(f'num_node_features is {data.num_node_features}')
		# print(f'data.pos is {data.pos}')
		# print(f'num_edges is {data.num_edges}')
		# print(f'edge_index is {data.edge_index}')
		# print(f'num_edge_features is {data.num_edge_features}')
		return data


# visualize_graph(data, 'g')

def construct_graph_and_visualize(f=0):
		G = nx.Graph()
		# 2. contruct the edge_index
		edge_index = create_edge_index_finger_p2t(f=f)
		# print(f'edge_list is {edge_index}')
		edge_list = [[edge_index[0][x], edge_index[1][x]] for x in range(edge_index.shape[1])]
		# print(f'edge_list is {edge_list}')
		G.add_edges_from(edge_list)

		fig, ax = plt.subplots(figsize=(8, 8))
		# option = {'font_family':"serif', 'font_size':'15', 'font_weight':'semibold'"}
		# nx.draw_networkx(G, node_size=400, **option)
		nx.draw_networkx(G, node_size=400)

		plt.axis('off')
		plt.show()


def visualize_whole():
		G = nx.Graph()
		# 2. contruct the edge_index
		edge_index = create_edge_index_whole_p2t()
		edge_list = [[edge_index[0][x], edge_index[1][x]] for x in range(edge_index.shape[1])]
		G.add_edges_from(edge_list)

		fig, ax = plt.subplots(figsize=(12, 12))
		# option = {'font_family':"serif', 'font_size':'15', 'font_weight':'semibold'"}
		# nx.draw_networkx(G, node_size=400, **option)
		nx.draw_networkx(G, node_size=400)

		plt.axis('off')
		plt.show()


def visualize_sec(f=4, s=2):
		G = nx.Graph()
		# 2. contruct the edge_index
		array = np.array([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]) + 16*f + 16*s
		sec_index_0 = array
		start, end = create_edge_single(sec_index_0)
		edge_index = np.array([start, end])
		edge_list = [[edge_index[0][x], edge_index[1][x]] for x in range(edge_index.shape[1])]
		G.add_edges_from(edge_list)

		fig, ax = plt.subplots(figsize=(12, 12))
		# option = {'font_family':"serif', 'font_size':'15', 'font_weight':'semibold'"}
		# nx.draw_networkx(G, node_size=400, **option)
		nx.draw_networkx(G, node_size=400)

		plt.axis('off')
		plt.show()


def visualize_sec_all():
		for i in range(12):
				plt.subplot(4, 3, i+1)
				G = nx.Graph()
				# 2. contruct the edge_index
				f = i // 3
				s = i-f*3
				print(f'{i}: finger {f} and section {s}')

				array = np.array([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]) + 48 * f + 16 * s
				print(array)
				sec_index_0 = array
				start, end = create_edge_single(sec_index_0)
				edge_index = np.array([start, end])
				edge_list = [[edge_index[0][x], edge_index[1][x]] for x in range(edge_index.shape[1])]
				G.add_edges_from(edge_list)

				# option = {'font_family':"serif', 'font_size':'15', 'font_weight':'semibold'"}
				# nx.draw_networkx(G, node_size=400, **option)
				nx.draw_networkx(G, node_size=400)

		plt.axis('off')
		plt.show()

# visualize_sec_all()


def visualize_finger(f=4):
		G = nx.Graph()
		# 2. contruct the edge_index

		edge_index = create_edge_index_finger_p2t(f)
		edge_list = [[edge_index[0][x], edge_index[1][x]] for x in range(edge_index.shape[1])]
		G.add_edges_from(edge_list)

		fig, ax = plt.subplots(figsize=(12, 12))
		# option = {'font_family':"serif', 'font_size':'15', 'font_weight':'semibold'"}
		# nx.draw_networkx(G, node_size=400, **option)
		nx.draw_networkx(G, node_size=400)

		plt.axis('off')
		plt.show()


def visualize_whole(f=4):
		G = nx.Graph()
		# 2. contruct the edge_index

		edge_index = create_edge_index_whole_p2t()
		# start, end = connect_within_palm()
		# edge_index = np.array([start, end])
		edge_list = [[edge_index[0][x], edge_index[1][x]] for x in range(edge_index.shape[1])]
		G.add_edges_from(edge_list)

		fig, ax = plt.subplots(figsize=(20, 20))
		# option = {'font_family':"serif', 'font_size':'15', 'font_weight':'semibold'"}
		# nx.draw_networkx(G, node_size=400, **option)
		nx.draw_networkx(G, node_size=400)

		plt.axis('off')
		plt.show()


visualize_whole()