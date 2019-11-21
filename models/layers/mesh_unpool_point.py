import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify

# TODO: Implement this
class MeshUnpoolPoint(nn.Module):
    def __init__(self, unroll_target, multi_thread=False):
        super(MeshUnpoolPoint, self).__init__()
        self.__out_target = unroll_target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]


    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)


    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        unpool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                unpool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                unpool_threads[-1].start()
            else:
                self.__unpool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                unpool_threads[mesh_index].join()
        fe = self.__updated_fe
        out_features = torch.cat(fe).view(len(meshes), -1, self.__out_target)
        return out_features


    def __unpool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.face_count], mesh.face_count)
        fe = self.__fe[mesh_index]
        edge_mask = np.ones(mesh.edges_count, dtype=np.bool)
        face_mask = np.ones(mesh.face_count, dtype=np.bool)
        # edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        init_face_count = mesh.face_count
        while mesh.face_count < self.__out_target:
            value, face_id = heappop(queue)
            face_id = int(face_id)
            neighbors = mesh.gemm_faces[face_id]
            if face_mask[face_id]:
                max_val = float("-inf")
                max_n = -1
                for n in neighbors:
                    if n<init_face_count:   #TODO: Right now we don't do unpooling if the neighbor is a new face. Consider including these cases
                        n_idx = np.where(np.asarray(queue)[:, 1] == n)[0]
                        if face_mask[n] and n_idx.size == 1:
                            val = queue[n_idx[0]][0]
                            if val > max_val:
                                max_val = val
                                max_n = n
                if max_n != -1:
                    edge_id = int(np.intersect1d(mesh.edges_in_face[face_id], mesh.edges_in_face[max_n])[0])
                    if edge_mask[edge_id]:
                        fe = self.__unpool_edge(mesh, edge_id, edge_mask, face_mask, fe, face_id, max_n, mesh_index)

        self.__updated_fe[mesh_index] = fe


    def __unpool_edge(self, mesh, edge_id, edge_mask, face_mask, fe, f1, f2, mesh_index):
        # Not pool if the edge or one of its neighbors is in a boundary
        if self.has_boundaries(mesh, edge_id):
            # TODO: We can include cases with boundaries, we just have to make sure not to create an edge in the neighbor
            return fe
        else:
            # TODO: Some of the generated faces are the wrong direction. Look into that
            # Find face vertices opposite to edge_id
            vt_f1 = mesh.faces[f1,mesh.faces[f1] != mesh.edges[edge_id,0]]
            vt_f1 = vt_f1[vt_f1 != mesh.edges[edge_id, 1]][0]
            vt_f2 = mesh.faces[f2, mesh.faces[f2] != mesh.edges[edge_id, 0]]
            vt_f2 = vt_f2[vt_f2 != mesh.edges[edge_id, 1]][0]

            # Divide edge_id in 2 edges
            vt_old = mesh.edges[edge_id][0]     # This is the vertex that remains in f1 and f2
            vt_new = mesh.edges[edge_id][1]     # This is the vertex that will be in new_f1 and new_f2
            mesh.vs = np.append(mesh.vs, [(mesh.vs[vt_old]+mesh.vs[mesh.edges[edge_id][1]])/2], axis=0)
            new_vt = mesh.vs.shape[0]-1
            mesh.edges = np.append(mesh.edges, [[new_vt, vt_new]], axis=0)
            mesh.edges_count += 1
            new_edge_id = mesh.edges.shape[0]-1
            mesh.edges[edge_id,1] = new_vt

            # Create new edges to divide the faces
            mesh.edges = np.append(mesh.edges, [[new_vt, vt_f1]], axis=0)
            mesh.edges_count += 1
            new_edge_f1 = mesh.edges.shape[0]-1
            mesh.edges = np.append(mesh.edges, [[new_vt, vt_f2]], axis=0)
            mesh.edges_count += 1
            new_edge_f2 = mesh.edges.shape[0] - 1

            mesh.faces = np.append(mesh.faces, [mesh.faces[f1]], axis=0)
            new_f1 = mesh.faces.shape[0]-1
            mesh.faces = np.append(mesh.faces, [mesh.faces[f2]], axis=0)
            new_f2 = mesh.faces.shape[0] - 1
            mesh.face_count+=2

            mesh.faces[f1, np.where(mesh.faces[f1]==vt_new)[0][0]] = new_vt
            mesh.faces[f2, np.where(mesh.faces[f2]==vt_new)[0][0]] = new_vt
            mesh.faces[new_f1, np.where(mesh.faces[new_f1] == vt_old)[0][0]] = new_vt
            mesh.faces[new_f2, np.where(mesh.faces[new_f2] == vt_old)[0][0]] = new_vt

            # Update neighbors
            n_f1 = mesh.gemm_faces[f1]
            n_f1 = n_f1[n_f1!=f2]
            if mesh.edges[edge_id,0] in mesh.faces[n_f1][0]:
                n_f1 = n_f1[1]
            elif mesh.edges[edge_id, 0] in mesh.faces[n_f1][1]:
                n_f1 = n_f1[0]
            else:
                assert(False)

            mesh.gemm_faces[f1, mesh.gemm_faces[f1] == n_f1] = new_f1
            mesh.gemm_faces = np.append(mesh.gemm_faces,[[n_f1,f1,new_f2]], axis=0)

            n_f2 = mesh.gemm_faces[f2]
            n_f2 = n_f2[n_f2 != f1]
            # n_f2 = n_f2[n_f2[np.where(mesh.edges[edge_id, 0] in mesh.faces[n_f2])[0]][0] != n_f2][0]
            if mesh.edges[edge_id,0] in mesh.faces[n_f2][0]:
                n_f2 = n_f2[1]
            elif mesh.edges[edge_id, 0] in mesh.faces[n_f2][1]:
                n_f2 = n_f2[0]
            else:
                assert(False)
            mesh.gemm_faces[f2, mesh.gemm_faces[f2] == n_f2] = new_f2
            mesh.gemm_faces = np.append(mesh.gemm_faces, [[n_f2, f2, new_f1]], axis=0)

            mesh.gemm_faces[n_f1, mesh.gemm_faces[n_f1] == f1] = new_f1
            mesh.gemm_faces[n_f2, mesh.gemm_faces[n_f2] == f2] = new_f2

            #Update edges_in_face
            vt_old = mesh.edges[edge_id, mesh.edges[edge_id]!=new_vt][0]

            edge_f1 = mesh.edges_in_face[f1, mesh.edges_in_face[f1]!=edge_id]
            assert(vt_old in mesh.edges[edge_f1])
            if vt_old in  mesh.edges[edge_f1[0]]:
                edge_new_f1 = edge_f1[1]
                edge_f1 = edge_f1[0]
            else:
                edge_new_f1 = edge_f1[0]
                edge_f1 = edge_f1[1]
            mesh.edges_in_face[f1] = np.array([edge_id,new_edge_f1,edge_f1])
            mesh.edges_in_face = np.append(mesh.edges_in_face, np.array([[new_edge_id,new_edge_f1,edge_new_f1]]),axis=0)

            edge_f2 = mesh.edges_in_face[f2, mesh.edges_in_face[f2] != edge_id]
            assert (vt_old in mesh.edges[edge_f2])
            if vt_old in mesh.edges[edge_f2[0]]:
                edge_new_f2 = edge_f2[1]
                edge_f2 = edge_f2[0]
            else:
                edge_new_f2 = edge_f2[0]
                edge_f2 = edge_f2[1]
            mesh.edges_in_face[f2] = np.array([edge_id, new_edge_f2, edge_f2])
            mesh.edges_in_face = np.append(mesh.edges_in_face, np.array([[new_edge_id, new_edge_f2, edge_new_f2]]),
                                           axis=0)
            
            #Features of new faces are the same as their "parent" face
            fe = torch.cat((fe, fe[:, f1].unsqueeze(1)), dim=1)
            fe = torch.cat((fe, fe[:, f2].unsqueeze(1)), dim=1)

            face_mask[f1] = False
            face_mask[f2] = False
            edge_mask[edge_id] = False

            return fe


    def __pool_face(self, mesh, face_mask, face_id, edge_id):
        # Remove face
        face_mask[face_id] = False
        mesh.face_count -= 1

        # Update neighbors
        neighbors = mesh.gemm_faces[face_id]
        neighbors = np.delete(neighbors, np.where(mesh.edges_in_face[neighbors] == edge_id)[0][0], axis=0)
        mesh.gemm_faces[neighbors[0], np.where(mesh.gemm_faces[neighbors[0]] == face_id)[0][0]] = neighbors[1]
        mesh.gemm_faces[neighbors[1], np.where(mesh.gemm_faces[neighbors[1]] == face_id)[0][0]] = neighbors[0]


    def __clean_side(self, mesh, edge_id, edge_mask, face_mask, edge_groups, side):
        if mesh.face_count <= self.__out_target:
            return False
        invalid_edges = MeshPoolFace.__get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.face_count > self.__out_target:
            self.__remove_triplete(mesh, edge_mask, face_mask, edge_groups, invalid_edges)
            if mesh.face_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True


    @staticmethod
    def has_boundaries(mesh, edge_id):
        # TODO: There is no boundaries in our initial mesh, but we may consider in the future the case where there is boundaries
        # for edge in mesh.gemm_edges[edge_id]:
        #     if edge == -1:
        #         return True
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
        mesh.edges_in_face[mesh.edges_in_face == key_b] = key_a
        MeshPoolFace.__remove_group(mesh, edge_groups, key_b)
        mesh.remove_edge(key_b)
        mesh.edges_count -= 1
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


    def __remove_triplete(self, mesh, edge_mask, face_mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            # print('Remove triplete in edge ', edge_key)
            vertex &= set(mesh.edges[edge_key])
            edge_mask[edge_key] = False
            MeshPoolFace.__remove_group(mesh, edge_groups, edge_key)

        # Get faces adjacent to vertex. Remove 2 and keep 1
        faces = np.where(mesh.faces == list(vertex)[0])[0]
        faces = faces[face_mask[faces]]
        assert len(faces) == 3
        face_mask[faces[1:]] = 0

        # Update neighbors of new face
        neighbors = mesh.gemm_faces[faces[1]]
        neighbors = neighbors[neighbors != faces[0]]
        n1 = neighbors[neighbors != faces[2]][0]
        mesh.gemm_faces[faces[0], np.where(mesh.gemm_faces[faces[0]] == faces[1])[0][0]] = n1
        mesh.gemm_faces[n1, np.where(mesh.gemm_faces[n1] == faces[1])[0][0]] = faces[0]

        neighbors = mesh.gemm_faces[faces[2]]
        neighbors = neighbors[neighbors != faces[0]]
        n2 = neighbors[neighbors != faces[1]][0]
        mesh.gemm_faces[faces[0], np.where(mesh.gemm_faces[faces[0]] == faces[2])[0][0]] = n2
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
        heap = torch.cat((-squared_magnitude, face_ids), dim=-1).tolist()
        heapify(heap, )
        return heap


    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        mesh.union_groups(source, target)


    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)