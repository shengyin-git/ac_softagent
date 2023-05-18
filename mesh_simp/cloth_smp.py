import numpy as np
import open3d as o3d
import pymeshlab
import time
import matplotlib.pyplot as plt 
import random 
from mpl_toolkits.mplot3d import Axes3D
# import pygeodesic.geodesic as geodesic
# import potpourri3d as pp3d 
# import gdist
import copy

class mesh_smp(object):
    def __init__(self):
        self.ms = pymeshlab.MeshSet()

    def import_flat_points(self, pos):
        self.flatten_ini_pos = pos       
        self.num_vertices = len(self.flatten_ini_pos)

    def generate_initial_mesh(self):
        m = pymeshlab.Mesh(self.flatten_ini_pos)

        self.ms.add_mesh(m, "ini_points")

        self.ms.surface_reconstruction_ball_pivoting(ballradius = pymeshlab.Percentage(3.2))

        _m = self.ms.mesh(0) #self.ms.current_mesh()
        self.ini_connections = _m.face_matrix()

    def generate_final_mesh(self, final_pos):
        m = pymeshlab.Mesh(final_pos, self.ini_connections)
        self.num_faces = m.face_number()
        self.ms.add_mesh(m, 'folded_mesh')

        # self.ms.set_current_mesh(0)

    def smp_the_mesh(self, threshold=5):
        ## simplify the mesh iteratively, best from the original number of triangles
        num_start = 2
        num_end = 200
        num_sample = np.int64(num_end - num_start) + 1
        simp_set = np.linspace(num_end, num_start, num_sample, dtype=np.int64)

        for i in range(num_sample):
            target_num_triangle = simp_set[i]
            tmp_str = 'simplified mesh '+str(num_end - i)
            if i == 0:
                m = self.ms.mesh(1)
                self.ms.add_mesh(m, tmp_str)
                self.ms.simplification_quadric_edge_collapse_decimation(targetfacenum = target_num_triangle, planarquadric = True, planarweight = 0.00001)
            else:
                m = self.ms.mesh(1+i)
                self.ms.add_mesh(m, tmp_str)
                self.ms.simplification_quadric_edge_collapse_decimation(targetfacenum = target_num_triangle, planarquadric = True, planarweight = 0.00001)

        # get rhe best simplification
        num_meshes = self.ms.number_meshes()

        not_improved = 0
        min_hausdorff = np.inf
        best_simp = 1        

        num_triangles = 2
        while not_improved < threshold:
            tmp_idx = num_meshes - num_triangles + 1
            output_1 = self.ms.hausdorff_distance(sampledmesh=1, targetmesh=tmp_idx, samplenum = self.num_vertices)
            output_2 = self.ms.hausdorff_distance(sampledmesh=tmp_idx, targetmesh=1, samplenum = self.num_vertices)
            hausdorff_dis = max(output_1['max'], output_2['max'])

            if hausdorff_dis > min_hausdorff or abs(hausdorff_dis - min_hausdorff)< 0.001:
                not_improved += 1
            else:
                min_hausdorff = hausdorff_dis
                not_improved = 0
                best_simp = tmp_idx
            num_triangles += 1

        self.best_simplification = best_simp

    def get_key_point(self):
        simp_m = self.ms.mesh(self.best_simplification)
        simp_vertices = simp_m.vertex_matrix()

        ori_m = self.ms.mesh(1)
        ori_vertices = ori_m.vertex_matrix()

        num_simp_vertices = len(simp_vertices)
        num_ori_vertices = len(ori_vertices)

        index = np.linspace(0, num_ori_vertices - 1, num_ori_vertices)

        key_point_index = [] 
        key_point_flat_pos = []
        simp_node_index = []
        
        for i in range(num_simp_vertices):
            distance = np.sqrt(np.sum(np.asarray(simp_vertices[i] - ori_vertices)**2, axis = 1))
            index_distance = np.transpose(np.vstack((index, distance))) 
            index_distance = index_distance[np.argsort(index_distance[:,1])]
            
            tmp_idx = np.int64(index_distance[0,0])
            simp_node_index.append(tmp_idx)

            if i == 0:                
                key_point_index.append(tmp_idx)
                key_point_flat_pos.append(self.flatten_ini_pos[tmp_idx])
                continue

            min_dis = min(np.sqrt(np.sum(np.asarray(self.flatten_ini_pos[tmp_idx] - key_point_flat_pos)**2, axis = 1)))  

            if min_dis > 0.05: 
                key_point_index.append(tmp_idx)
                key_point_flat_pos.append(self.flatten_ini_pos[tmp_idx])

        self.key_point_index = np.array(key_point_index, dtype=np.int64)
        return self.key_point_index 

    def plot_simplified_mesh(self):
        fig = plt.figure(figsize=(10,10))
        ax = Axes3D(fig)
        fig.add_axes(ax)

        ori_m = self.ms.mesh(1)
        ori_vertices = ori_m.vertex_matrix()
        ori_triangles = ori_m.face_matrix()
        #EE82EE
        ax.scatter(ori_vertices[:,0], ori_vertices[:,2], ori_vertices[:,1],marker = 'o', c='#FF82AB', s=10, alpha = 0.4, label = 'Original mesh') #, label='original point position'
        # ax.scatter(ori_vertices[self.key_point_index,0], ori_vertices[self.key_point_index,2], ori_vertices[self.key_point_index,1], marker = '<', s = 2000, c='b', label='Extracted key points')

        ori_vertices = self.flatten_ini_pos
        # ax.scatter(ori_vertices[:,0], ori_vertices[:,2], ori_vertices[:,1],marker = '<', c='r', s=50, alpha = 0.2, label = 'Original mesh') #, label='original point position'

        simp_m = self.ms.mesh(self.best_simplification)
        simp_vertices = simp_m.vertex_matrix()
        simp_triangles = simp_m.face_matrix()

        num_triangles = len(simp_triangles)
        linewidth = np.flip(np.linspace(1,num_triangles*2, num_triangles))

        num_temp_1 = np.linspace(10,255, num_triangles)
        num_temp_2 = np.linspace(10,255, num_triangles)
        num_temp_3 = np.linspace(10,255, num_triangles)
        random.shuffle(num_temp_1)
        random.shuffle(num_temp_2)
        random.shuffle(num_temp_3)        
        color = np.vstack((num_temp_1, num_temp_2, num_temp_3)).transpose()/255

        for i in range(num_triangles): 

            index_0 = simp_triangles[i]
            index_1 = np.hstack([simp_triangles[i], simp_triangles[i][0]])

            x_0 = simp_vertices[index_0, 0]
            y_0 = simp_vertices[index_0, 2]
            z_0 = simp_vertices[index_0, 1]

            x_1 = simp_vertices[index_1, 0]
            y_1 = simp_vertices[index_1, 2]
            z_1 = simp_vertices[index_1, 1]
            #00FFFF
            ax.plot(x_1, y_1, z_1, color = '#00F5FF', linestyle='--', linewidth = 5, markersize = 30, marker = 'o')
            ax.plot_trisurf(x_0, y_0, z_0, color = '#00F5FF', alpha = 0.5) 

        ax.axis('off')

        ax.set_zlim3d(0,0.15)
        plt.show()

def main():
    pass

if __name__ == '__main__':
    main()
        

