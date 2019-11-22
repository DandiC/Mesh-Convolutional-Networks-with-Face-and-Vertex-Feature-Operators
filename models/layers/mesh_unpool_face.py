import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify

# TODO: Implement this
class MeshUnpoolFace(nn.Module):
    def __init__(self, unroll_target, multi_thread=False):
        super(MeshUnpoolFace, self).__init__()
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

    @staticmethod
    def has_boundaries(mesh, edge_id):
        # TODO: There is no boundaries in our initial mesh, but we may consider in the future the case where there is boundaries
        # for edge in mesh.gemm_edges[edge_id]:
        #     if edge == -1:
        #         return True
        return False

    def __build_queue(self, features, face_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        face_ids = torch.arange(face_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((-squared_magnitude, face_ids), dim=-1).tolist()
        heapify(heap, )
        return heap