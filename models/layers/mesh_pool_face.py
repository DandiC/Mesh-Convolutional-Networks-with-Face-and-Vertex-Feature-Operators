import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify


class MeshPoolFace(nn.Module):

    def __init__(self, target, multi_thread=False):
        super(MeshPoolFace, self).__init__()
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
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.face_count], mesh.face_count)
        # recycle = []
        # last_queue_len = len(queue)
        last_count = mesh.face_count + 1
        mask = np.ones(mesh.face_count, dtype=np.uint8)
        edge_groups = MeshUnion(mesh.face_count, self.__fe.device)
        fe = self.__fe[mesh_index, :, :mesh.face_count, :]
        # print('Pooling mesh ', mesh.filename)
        while mesh.face_count > self.__out_target:
            value, face_id = heappop(queue)
            face_id = int(face_id)
            if face_id != -1:
                fe, queue = self.__pool_face(mesh, face_id, fe, queue)

        # print('')
        # mesh.clean(mask, edge_groups)
        # fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe

    def __pool_face(self, mesh, face_id, fe, queue):
        # Modify mesh to debug
        # mesh.vs = np.random.rand(7, 3)
        # mesh.faces = np.array([[0, 1, 5], [1, 2, 5], [2, 5, 6], [2, 3, 6], [0, 6, 3], [0, 5, 6], [0, 3, 4], [1, 0, 4]])
        # mesh.gemm_faces = np.array(
        #     [[1, 5, 7], [2, 0, -1], [1, 3, 5], [2, 4, -1], [3, 6, 5], [2, 4, 0], [4, 7, -1], [0, 6, -1]])
        # face_id = 5
        # queue = [[0.001, 3], [0.002, 1], [0.003, 2], [0.004, 0], [0.005, 7], [0.006, 6], [0.05, 4]]
        # mesh.face_count = 8
        # mesh.face_areas = np.random.rand(mesh.face_count,)
        # fe = fe[:,:mesh.face_count,:]
        assert np.where(np.unique(mesh.gemm_faces, return_counts=True)[1] != 3)[0].size == 0
        if self.has_boundaries(mesh, face_id):
            return fe, queue
        else:
            # Compute center as average between 3 points
            center = np.mean(mesh.vs[mesh.faces[face_id]], axis=0)
            # Move first vertex (arbitrary) towards the center

            mesh.vs[mesh.faces[face_id, 0]] = center
            # Delete other vertices and make their faces point to "center"
            for i in range(1, mesh.faces[face_id].shape[0]):
                v_id = mesh.faces[face_id, i]
                mesh.faces[np.where(mesh.faces == v_id)] = mesh.faces[face_id, 0]
                mesh.v_mask[v_id] = False

            # Set new neighbors
            # print(mesh.gemm_faces[mesh.gemm_faces[face_id]])
            # print(face_id)
            for face in mesh.gemm_faces[face_id]:
                # print('    ', face)
                # Get neighbors of the neighbor (ommiting the selected face)
                neighbors = mesh.gemm_faces[face]
                # print('      neighbors:', neighbors)
                neighbors = neighbors[neighbors != face_id]
                assert len(neighbors) == 2
                # print(len(neighbors))
                # Set new neighbors (only for trimeshes)
                # print('        neighbors of', neighbors[0],':',mesh.gemm_faces[neighbors[0]])
                # print('        neighbors of', neighbors[1], ':', mesh.gemm_faces[neighbors[1]])
                # print('In pool: ', mesh.gemm_faces[neighbors[0]], mesh.gemm_faces[neighbors[1]])
                assert np.where(mesh.gemm_faces[neighbors[0]]==neighbors[1])[0].size==0 and np.where(mesh.gemm_faces[neighbors[1]]==neighbors[0])[0].size==0
                # TODO: Extend for other types of mesh
                mesh.gemm_faces[neighbors[0], np.where(mesh.gemm_faces[neighbors[0]] == face)[0][0]] = neighbors[1]
                mesh.gemm_faces[neighbors[1], np.where(mesh.gemm_faces[neighbors[1]] == face)[0][0]] = neighbors[0]

            assert all(mesh.gemm_faces[:, 0] != mesh.gemm_faces[:, 1]) and all(
                mesh.gemm_faces[:, 1] != mesh.gemm_faces[:, 2]) and all(mesh.gemm_faces[:, 0] != mesh.gemm_faces[:, 2])
            # Delete face and its neighbors
            todelete = np.concatenate(([face_id], mesh.gemm_faces[face_id]))
            npqueue = np.asarray(queue)
            # mask = np.ones(npqueue.shape[0], dtype=np.uint8)

            # Delete elements from queue (only neighbors, face_id is already out)
            for f in todelete[1:]:
                npqueue[npqueue[:,1]==f, 1] =-1
            # assert np.where(mask == 0)[0].size == 3
            # npqueue[mask == 0, 1] = -1
            mask = np.ones(mesh.face_count, dtype=np.uint8)
            mask[todelete] = 0

            # Delete faces
            mesh.faces = mesh.faces[mask == 1, :]
            mesh.face_areas = mesh.face_areas[mask == 1]
            mesh.gemm_faces = mesh.gemm_faces[mask == 1, :]


            # Update references to face ids in gemm and queue
            todelete.sort()
            for f in todelete[::-1]:
                mesh.gemm_faces[np.where(mesh.gemm_faces >= f)] -= 1
                npqueue[np.where(npqueue[:, 1] >= f), 1] -= 1


            # Update face count
            mesh.face_count -= todelete.shape[0]

            # Delete features
            fe = fe[:, np.where(mask == 1)]
            # print(todelete)

            return torch.reshape(fe, [fe.shape[0], fe.shape[2], fe.shape[3]]), npqueue.tolist()

        # elif self.__clean_side(mesh, face_id, mask, edge_groups, 0)\
        #     and self.__clean_side(mesh, face_id, mask, edge_groups, 2) \
        #     and self.__is_one_ring_valid(mesh, face_id):
        #     self.__merge_edges[0] = self.__pool_side(mesh, face_id, mask, edge_groups, 0)
        #     self.__merge_edges[1] = self.__pool_side(mesh, face_id, mask, edge_groups, 2)
        #     mesh.merge_vertices(face_id)
        #     mask[face_id] = False
        #     MeshPoolFace.__remove_group(mesh, edge_groups, face_id)
        #     mesh.face_count -= 1
        #     return True
        # else:
        #     return False

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.face_count <= self.__out_target:
            return False
        invalid_edges = MeshPoolFace.__get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.face_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.face_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True

    @staticmethod
    def has_boundaries(mesh, face_id):
        prev_neighbors = []
        for face in mesh.gemm_faces[face_id]:
            if face == -1 or -1 in mesh.gemm_faces[face]:
                return True
            if face in mesh.gemm_faces[mesh.gemm_faces[face_id]]:
                return True
            neighbors = mesh.gemm_faces[face]
            neighbors = neighbors[neighbors != face_id]
            if neighbors[0] in prev_neighbors or neighbors[1] in prev_neighbors:
                return True
            prev_neighbors.append(neighbors[0])
            prev_neighbors.append(neighbors[1])
            # print('In has_boundaries: ', mesh.gemm_faces[neighbors[0]], mesh.gemm_faces[neighbors[1]])
            if not (np.where(mesh.gemm_faces[neighbors[0]]==neighbors[1])[0].size==0 and np.where(mesh.gemm_faces[neighbors[1]]==neighbors[0])[0].size==0):
                return True
        return False

    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPoolFace.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1],
                              mesh.sides[key_b, other_side_b + 1])
        MeshPoolFace.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPoolFace.__union_groups(mesh, edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPoolFace.__remove_group(mesh, edge_groups, key_b)
        mesh.remove_edge(key_b)
        mesh.face_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPoolFace.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        shared_items = MeshPoolFace.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert (len(shared_items) == 2)
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            MeshPoolFace.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPoolFace.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b)
            MeshPoolFace.__redirect_edges(mesh, update_key_a, MeshPoolFace.__get_other_side(update_side_a),
                                          update_key_b, MeshPoolFace.__get_other_side(update_side_b))
            MeshPoolFace.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPoolFace.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPoolFace.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPoolFace.__union_groups(mesh, edge_groups, middle_edge, update_key_a)
            MeshPoolFace.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPoolFace.__union_groups(mesh, edge_groups, middle_edge, update_key_b)
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
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPoolFace.__remove_group(mesh, edge_groups, edge_key)
        mesh.face_count -= 3
        vertex = list(vertex)
        assert (len(vertex) == 1)
        mesh.remove_vertex(vertex[0])

    def __build_queue(self, features, face_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        face_ids = torch.arange(face_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, face_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)
