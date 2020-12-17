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
        fe = self.__fe[mesh_index]
        if mesh.face_count <= self.__out_target:
            self.__updated_fe[mesh_index] = fe[:, :self.__out_target]
            return

        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.face_count], mesh.face_count)
        orig_queue = queue.copy()

        edge_mask = np.ones(mesh.edges_count, dtype=np.bool)
        face_mask = np.ones(mesh.face_count, dtype=np.bool)
        face_groups = MeshUnion(mesh.face_count, self.__fe.device)

        while mesh.face_count > self.__out_target:
            if queue==[]:
                print('Run out of faces to pool')
                print(' Mesh:', mesh.filename)
                print(' # of current faces', mesh.face_count)
                print(' Target:', self._MeshPoolFace__out_target)

            value, face_id = heappop(queue)
            face_id = int(face_id)
            neighbors = mesh.gemm_faces[face_id]
            if face_mask[face_id]:
                min_val = float("inf")
                min_n = -1
                for n in neighbors:
                    n_idx = np.where(np.asarray(orig_queue)[:, 1] == n)[0]
                    if face_mask[n] and n_idx.size == 1:
                        val = orig_queue[n_idx[0]][0]
                        if val < min_val:
                            min_val = val
                            min_n = n
                if min_n != -1:
                    edge_id = int(np.intersect1d(mesh.edges_in_face[face_id], mesh.edges_in_face[min_n])[0])
                    if edge_mask[edge_id]:
                        self.__pool_edge(mesh, edge_id, edge_mask, face_mask, face_groups, face_id, min_n)
        mesh.cleanWithFace(edge_mask, face_mask, face_groups)
        fe = face_groups.rebuild_features(self.__fe[mesh_index], face_mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe
        # self.__updated_fe[mesh_index] = fe[:, face_mask]

    def __pool_edge(self, mesh, edge_id, edge_mask, face_mask, face_groups, f1, f2):
        # Not pool if the edge or one of its neighbors is in a boundary
        if self.has_boundaries(mesh, edge_id):
            return False
        elif self.__clean_side(mesh, edge_id, edge_mask, face_mask, face_groups, 0) \
                and self.__clean_side(mesh, edge_id, edge_mask, face_mask, face_groups, 2) \
                and self.__is_one_ring_valid(mesh, edge_id):

            # Merge side edges
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, edge_mask, face_groups, 0)
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, edge_mask, face_groups, 2)
            _, _ = mesh.merge_vertices(edge_id)

            # Remove edge
            edge_mask[edge_id] = False
            mesh.edges_count -= 1

            # Remove faces
            self.__pool_face(mesh, face_mask, f1, edge_id, face_groups)
            self.__pool_face(mesh, face_mask, f2, edge_id, face_groups)

            return True
        else:
            return False

    def __pool_face(self, mesh, face_mask, face_id, edge_id, face_groups):
        # Remove face
        face_mask[face_id] = False
        mesh.face_count -= 1

        # Update neighbors
        neighbors = mesh.gemm_faces[face_id]
        neighbors = np.delete(neighbors, np.where(mesh.edges_in_face[neighbors] == edge_id)[0][0], axis=0)
        mesh.gemm_faces[neighbors[0], np.where(mesh.gemm_faces[neighbors[0]] == face_id)[0][0]] = neighbors[1]
        mesh.gemm_faces[neighbors[1], np.where(mesh.gemm_faces[neighbors[1]] == face_id)[0][0]] = neighbors[0]
        MeshPoolFace.__union_groups(mesh, face_groups, face_id, neighbors[0])
        MeshPoolFace.__union_groups(mesh, face_groups, face_id, neighbors[1])
        MeshPoolFace.__remove_group(mesh, face_groups, face_id)

    def __clean_side(self, mesh, edge_id, edge_mask, face_mask, face_groups, side):
        if mesh.face_count <= self.__out_target:
            return False
        invalid_edges = MeshPoolFace.__get_invalids(mesh, edge_id, face_groups, side)
        while len(invalid_edges) != 0 and mesh.face_count > self.__out_target:
            self.__remove_triplete(mesh, edge_mask, face_mask, face_groups, invalid_edges)
            if mesh.face_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, face_groups, side)
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

    def __pool_side(self, mesh, edge_id, mask, face_groups, side):
        info = MeshPoolFace.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1],
                              mesh.sides[key_b, other_side_b + 1])
        mask[key_b] = False
        mesh.edges_in_face[mesh.edges_in_face == key_b] = key_a
        mesh.remove_edge(key_b)
        mesh.edges_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, face_groups, side):
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

    def __remove_triplete(self, mesh, edge_mask, face_mask, face_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            # print('Remove triplete in edge ', edge_key)
            vertex &= set(mesh.edges[edge_key])
            edge_mask[edge_key] = False

        # Get faces adjacent to vertex. Remove 2 and keep 1
        faces = set()
        for edge_key in invalid_edges:
            faces |= set(np.where(mesh.edges_in_face == edge_key)[0])
        faces = np.array(list(faces))
        faces = faces[face_mask[faces]]
        assert len(faces) == 3
        face_mask[faces[1:]] = 0

        MeshPoolFace.__union_groups(mesh, face_groups, faces[1], faces[0])
        MeshPoolFace.__union_groups(mesh, face_groups, faces[2], faces[0])
        MeshPoolFace.__remove_group(mesh, face_groups, faces[1])
        MeshPoolFace.__remove_group(mesh, face_groups, faces[2])

        # Update neighbors of new face
        neighbors = mesh.gemm_faces[faces[1]]
        neighbors = neighbors[neighbors != faces[0]]
        n1 = neighbors[neighbors != faces[2]][0]
        mesh.gemm_faces[faces[0], np.where(mesh.gemm_faces[faces[0]] == faces[1])[0][0]] = n1
        if n1 != -1:
            mesh.gemm_faces[n1, np.where(mesh.gemm_faces[n1] == faces[1])[0][0]] = faces[0]


        neighbors = mesh.gemm_faces[faces[2]]
        neighbors = neighbors[neighbors != faces[0]]
        n2 = neighbors[neighbors != faces[1]][0]
        mesh.gemm_faces[faces[0], np.where(mesh.gemm_faces[faces[0]] == faces[2])[0][0]] = n2
        if n2 != -1:
            mesh.gemm_faces[n2, np.where(mesh.gemm_faces[n2] == faces[2])[0][0]] = faces[0]

        # Udate edges in face of new face
        edges = mesh.edges_in_face[faces[0]]
        for i, edge in enumerate(edges):
            if edge in invalid_edges:
                if edge in mesh.edges_in_face[faces[1]]:
                    neighbors = mesh.edges_in_face[faces[1]]
                elif edge in mesh.edges_in_face[faces[2]]:
                    neighbors = mesh.edges_in_face[faces[2]]

                for invalid in invalid_edges:
                    neighbors = neighbors[neighbors != invalid]
                mesh.edges_in_face[faces[0], i] = neighbors[0]

        mesh.face_count -= 2
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert (len(vertex) == 1)
        mesh.remove_vertex(vertex[0])
        vertices = np.reshape(mesh.edges[invalid_edges], -1)
        for v in mesh.faces[faces[0]]:
            vertices = vertices[vertices != v]
        assert len(vertices) == 1
        mesh.faces[faces[0], np.where(mesh.faces[faces[0]] == vertex[0])[0][0]] = vertices[0]

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
    def __union_groups(mesh, face_groups, source, target):
        face_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, face_groups, index):
        face_groups.remove_group(index)
        mesh.remove_group(index)

