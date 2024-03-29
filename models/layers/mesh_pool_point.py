import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify
from torch.nn import ConstantPad2d

class MeshPoolPoint(nn.Module):
    
    def __init__(self, target, multi_thread=False):
        super(MeshPoolPoint, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        # ufe = self.__updated_fe
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        fe = self.__fe[mesh_index]
        if mesh.vs_count <= self.__out_target:
            self.__updated_fe[mesh_index] = fe[:, :self.__out_target]
            return

        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.vs_count], mesh)
        # recycle = []
        # last_queue_len = len(queue)
        # last_count = mesh.edges_count + 1
        edge_mask = np.ones(mesh.edges_count, dtype=np.bool)
        vertex_groups = MeshUnion(mesh.vs_count, self.__fe.device)
        while mesh.vs_count > self.__out_target:
            if not queue:
                print('Run out of vertices to pool')
                print(' Mesh:', mesh.filename)
                print(' # of current vertices:', mesh.vs_count)
                print(' Target:', self.__out_target)

            value, vt_id, n_id = heappop(queue)
            vt_id = int(vt_id)
            n_id = int(n_id)

            if mesh.v_mask[vt_id] and mesh.v_mask[n_id]:
                edge_id = np.argmax(np.logical_or(np.logical_and(mesh.edges[:, 0] == vt_id, mesh.edges[:, 1] == n_id),
                                                  np.logical_and(mesh.edges[:, 0] == n_id, mesh.edges[:, 1] == vt_id)))
                if edge_mask[edge_id]:
                    self.__pool_edge(mesh, edge_id, edge_mask, vertex_groups)
                    assert(mesh.vs_count == mesh.v_mask.sum())

        # Copy vertex mask so that it can be used when rebuilding the features
        v_mask = mesh.v_mask.copy()
        mesh.cleanWithPoint(edge_mask, vertex_groups)

        fe = vertex_groups.rebuild_features(self.__fe[mesh_index], v_mask, self.__out_target)

        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, vertex_groups):
        # Not pool if the edge or one of its neighbors is in a boundary
        if self.has_boundaries(mesh, edge_id):
            return False
        elif self.__clean_side(mesh, edge_id, mask, vertex_groups, 0)\
            and self.__clean_side(mesh, edge_id, mask, vertex_groups, 2) \
            and self.__is_one_ring_valid(mesh, edge_id):
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask, vertex_groups, 0)
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask, vertex_groups, 2)
            # Keeps the first component of edge_id (updated accordingly) and deletes the second.
            updated_v, removed_v = mesh.merge_vertices(edge_id)
            mask[edge_id] = False
            MeshPoolPoint.__union_groups(mesh, vertex_groups, removed_v, updated_v)
            MeshPoolPoint.__remove_group(mesh, vertex_groups, removed_v)
            mesh.edges_count -= 1
            return True
        else:
            return False

    def __clean_side(self, mesh, edge_id, mask, vertex_groups, side):
        if mesh.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPoolPoint.__get_invalids(mesh, edge_id, vertex_groups, side)
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, vertex_groups, invalid_edges)
            if mesh.vs_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, vertex_groups, side)
        return True

    @staticmethod
    def has_boundaries(mesh, edge_id):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False


    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, vertex_groups, side):
        info = MeshPoolPoint.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1], mesh.sides[key_b, other_side_b + 1])
        mask[key_b] = False
        mesh.remove_edge(key_b)
        mesh.edges_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, vertex_groups, side):
        info = MeshPoolPoint.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        shared_items = MeshPoolPoint.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert (len(shared_items) == 2)
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            MeshPoolPoint.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPoolPoint.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b)
            MeshPoolPoint.__redirect_edges(mesh, update_key_a, MeshPoolPoint.__get_other_side(update_side_a), update_key_b, MeshPoolPoint.__get_other_side(update_side_b))
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b

    @staticmethod
    def __remove_triplete(mesh, mask, vertex_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        vertices = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            vertices |= set(mesh.edges[edge_key])
            mask[edge_key] = False
        vertices = vertices.difference(vertex)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert(len(vertex) == 1)
        mesh.remove_vertex(vertex[0])
        for neighbor in vertices:
            MeshPoolPoint.__union_groups(mesh, vertex_groups, vertex[0], neighbor)
        MeshPoolPoint.__remove_group(mesh, vertex_groups, vertex[0])

    def __build_queue(self, features, mesh):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        heap = []
        pairs_in_heap = set()
        for i in range(mesh.vs_count):
            for n in list(mesh.gemm_vs[i]):
                if (i, n) not in pairs_in_heap:
                    m = (squared_magnitude[i, 0].data + squared_magnitude[n, 0].data).tolist()
                    heap.append([m, i, n])
                    pairs_in_heap.add((i, n))
                    pairs_in_heap.add((n, i))

        heapify(heap, )
        return heap

    @staticmethod
    def __union_groups(mesh, vertex_groups, source, target):
        vertex_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, vertex_groups, index):
        vertex_groups.remove_group(index)
        mesh.remove_group(index)

