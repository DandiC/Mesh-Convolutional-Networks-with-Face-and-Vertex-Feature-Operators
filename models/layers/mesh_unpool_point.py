import torch
import torch.nn as nn
from threading import Thread
import numpy as np
from heapq import heappop, heapify


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
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.vs_count], mesh)
        fe = self.__fe[mesh_index]
        edge_mask = np.ones(mesh.edges_count, dtype=np.bool)
        face_mask = np.ones(mesh.face_count, dtype=np.bool)
        # edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        init_vs_count = mesh.vs_count
        while mesh.vs_count < self.__out_target:
            value, vt_id, n_id = heappop(queue)
            vt_id = int(vt_id)
            n_id = int(n_id)

            edge_id = np.argmax(np.logical_or(np.logical_and(mesh.edges[:, 0] == vt_id, mesh.edges[:, 1] == n_id),
                                              np.logical_and(mesh.edges[:, 0] == n_id, mesh.edges[:, 1] == vt_id)))
            faces = np.argwhere(np.any(mesh.edges_in_face == edge_id, axis=1))[:, 0]
            assert (len(faces) == 2)
            if edge_mask[edge_id]:
                fe = self.__unpool_edge(mesh, edge_id, edge_mask, face_mask, fe, faces[0], faces[1])

        self.__updated_fe[mesh_index] = fe

    def __unpool_edge(self, mesh, edge_id, edge_mask, face_mask, fe, f1, f2):
        # Not pool if the edge or one of its neighbors is in a boundary
        if self.has_boundaries(mesh, edge_id):
            # TODO: We can include cases with boundaries, we just have to make sure not to create an edge in the neighbor
            return fe
        else:
            # Find face vertices opposite to edge_id
            vt_f1 = mesh.faces[f1, mesh.faces[f1] != mesh.edges[edge_id, 0]]
            vt_f1 = vt_f1[vt_f1 != mesh.edges[edge_id, 1]][0]
            vt_f2 = mesh.faces[f2, mesh.faces[f2] != mesh.edges[edge_id, 0]]
            vt_f2 = vt_f2[vt_f2 != mesh.edges[edge_id, 1]][0]

            # Divide edge_id in 2 edges
            vt_old = mesh.edges[edge_id][0]  # This is the vertex that remains in f1 and f2
            vt_new = mesh.edges[edge_id][1]  # This is the vertex that will be in new_f1 and new_f2
            mesh.vs = np.append(mesh.vs, [(mesh.vs[vt_old] + mesh.vs[mesh.edges[edge_id][1]]) / 2], axis=0)
            mesh.vs_count += 1
            new_vt = mesh.vs.shape[0] - 1
            mesh.edges = np.append(mesh.edges, [[new_vt, vt_new]], axis=0)
            mesh.edges_count += 1
            new_edge_id = mesh.edges.shape[0] - 1
            mesh.edges[edge_id, 1] = new_vt

            # Update gemm_vs
            mesh.gemm_vs[vt_old] = set([x if (x!=vt_new) else new_vt for x in mesh.gemm_vs[vt_old]])
            mesh.gemm_vs[vt_new] = set([x if (x != vt_old) else new_vt for x in mesh.gemm_vs[vt_new]])
            mesh.gemm_vs[vt_f1].add(new_vt)
            mesh.gemm_vs[vt_f2].add(new_vt)
            mesh.gemm_vs = np.append(mesh.gemm_vs, [set([vt_old, vt_new, vt_f1, vt_f2])], axis=0)

            # Create new edges to divide the faces
            mesh.edges = np.append(mesh.edges, [[new_vt, vt_f1]], axis=0)
            mesh.edges_count += 1
            new_edge_f1 = mesh.edges.shape[0] - 1
            mesh.edges = np.append(mesh.edges, [[new_vt, vt_f2]], axis=0)
            mesh.edges_count += 1
            new_edge_f2 = mesh.edges.shape[0] - 1

            mesh.faces = np.append(mesh.faces, [mesh.faces[f1]], axis=0)
            new_f1 = mesh.faces.shape[0] - 1
            mesh.faces = np.append(mesh.faces, [mesh.faces[f2]], axis=0)
            new_f2 = mesh.faces.shape[0] - 1
            mesh.face_count += 2

            mesh.faces[f1, np.where(mesh.faces[f1] == vt_new)[0][0]] = new_vt
            mesh.faces[f2, np.where(mesh.faces[f2] == vt_new)[0][0]] = new_vt
            mesh.faces[new_f1, np.where(mesh.faces[new_f1] == vt_old)[0][0]] = new_vt
            mesh.faces[new_f2, np.where(mesh.faces[new_f2] == vt_old)[0][0]] = new_vt

            # Update neighbors
            n_f1 = mesh.gemm_faces[f1]
            n_f1 = n_f1[n_f1 != f2]
            if mesh.edges[edge_id, 0] in mesh.faces[n_f1][0]:
                n_f1 = n_f1[1]
            elif mesh.edges[edge_id, 0] in mesh.faces[n_f1][1]:
                n_f1 = n_f1[0]
            else:
                assert (False)

            mesh.gemm_faces[f1, mesh.gemm_faces[f1] == n_f1] = new_f1
            mesh.gemm_faces = np.append(mesh.gemm_faces, [[n_f1, f1, new_f2]], axis=0)

            n_f2 = mesh.gemm_faces[f2]
            n_f2 = n_f2[n_f2 != f1]
            # n_f2 = n_f2[n_f2[np.where(mesh.edges[edge_id, 0] in mesh.faces[n_f2])[0]][0] != n_f2][0]
            if mesh.edges[edge_id, 0] in mesh.faces[n_f2][0]:
                n_f2 = n_f2[1]
            elif mesh.edges[edge_id, 0] in mesh.faces[n_f2][1]:
                n_f2 = n_f2[0]
            else:
                assert (False)
            mesh.gemm_faces[f2, mesh.gemm_faces[f2] == n_f2] = new_f2
            mesh.gemm_faces = np.append(mesh.gemm_faces, [[n_f2, f2, new_f1]], axis=0)

            mesh.gemm_faces[n_f1, mesh.gemm_faces[n_f1] == f1] = new_f1
            mesh.gemm_faces[n_f2, mesh.gemm_faces[n_f2] == f2] = new_f2

            # Update edges_in_face
            vt_old = mesh.edges[edge_id, mesh.edges[edge_id] != new_vt][0]

            edge_f1 = mesh.edges_in_face[f1, mesh.edges_in_face[f1] != edge_id]
            assert (vt_old in mesh.edges[edge_f1])
            if vt_old in mesh.edges[edge_f1[0]]:
                edge_new_f1 = edge_f1[1]
                edge_f1 = edge_f1[0]
            else:
                edge_new_f1 = edge_f1[0]
                edge_f1 = edge_f1[1]
            mesh.edges_in_face[f1] = np.array([edge_id, new_edge_f1, edge_f1])
            mesh.edges_in_face = np.append(mesh.edges_in_face, np.array([[new_edge_id, new_edge_f1, edge_new_f1]]),
                                           axis=0)

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

            # Features of new vertex is the average between the two parents
            fe = torch.cat((fe,torch.mean(fe[:,[vt_old,vt_new]],axis=1).unsqueeze(1)),dim=1)

            # face_mask[f1] = False
            # face_mask[f2] = False
            edge_mask[edge_id] = False

            return fe

    @staticmethod
    def has_boundaries(mesh, edge_id):
        # TODO: There is no boundaries in our initial mesh, but we may consider in the future the case where there is boundaries
        # for edge in mesh.gemm_edges[edge_id]:
        #     if edge == -1:
        #         return True
        return False

    def __build_queue(self, features, mesh):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        heap = []
        pairs_in_heap = {}
        for i in range(mesh.vs_count):
            for n in list(mesh.gemm_vs[i]):
                if (i,n) not in pairs_in_heap:
                    m = (squared_magnitude[i,0].data+squared_magnitude[n,0].data).tolist()
                    heap.append([-m,i,n])
                    pairs_in_heap[(i,n)] = True
                    pairs_in_heap[(n, i)] = True

        heapify(heap, )
        return heap
