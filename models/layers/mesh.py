from tempfile import mkstemp
from shutil import move
import torch
import numpy as np
import os
from models.layers.mesh_union import MeshUnion
from models.layers.mesh_prepare import fill_mesh, build_gemm_vs


class Mesh:

    def __init__(self, file=None, opt=None, hold_history=False, export_folder='', faces=None, vertices=None,
                 feat_from='face', export_filename='unknown'):
        if opt != None:
            feat_from = opt.feat_from
        self.opt = opt
        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        self.edges = self.gemm_edges = self.sides = None
        self.pool_count = self.face_count = 0
        self.faces = self.face_areas = self.gemm_faces = None
        # Extracts information from OBJ meshes (vertices, edges, features etc.)
        fill_mesh(self, file, opt, faces=faces, vertices=vertices, feat_from=feat_from)
        self.export_folder = export_folder
        self.history_data = None
        if hold_history:
            self.init_history()
        self.export()

    def extract_features(self):
        return self.features

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        to_keep, to_remove = edge
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.remove_vertex(edge[1])
        for index, vt in enumerate(self.gemm_vs_raw[edge[1]]):
            if vt != edge[0]:
                self.gemm_vs_raw[vt].add(edge[0])
                self.gemm_vs_raw[edge[0]].add(vt)

        self.faces[self.faces == edge[1]] = edge[0]
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]
        # Return the vertex index that stays, and the one that is removed.
        return to_keep, to_remove

    def remove_vertex(self, v):
        if self.v_mask[v]:
            self.vs_count -= 1
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id in self.ve[v]:
                self.ve[v].remove(edge_id)

    # TODO: Combine clean, cleanWithPoint and cleanWithFace into one.
    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()

    def cleanWithPoint(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(self.v_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]

        self.vs = self.vs[self.v_mask]

        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]

        for v_index, ve in enumerate(self.ve):
            update_ve = []
            if self.v_mask[v_index]:
                for e in ve:
                    update_ve.append(new_indices[e])
                new_ve.append(update_ve)
        self.ve = new_ve

        v_mask = np.concatenate([self.v_mask, [False]])
        new_vs_indices = np.zeros(v_mask.shape[0], dtype=np.int32)
        new_vs_indices[-1] = -1
        new_vs_indices[v_mask] = np.arange(0, np.ma.where(v_mask)[0].shape[0])
        self.edges = new_vs_indices[self.edges[:, :]]
        new_gemm_vs = [set() for _ in range(self.vs.shape[0])]
        for v_index, gemm in enumerate(self.gemm_vs_raw):
            if self.v_mask[v_index]:
                for vt in gemm:
                    if self.v_mask[vt]:
                        new_gemm_vs[new_vs_indices[v_index]].add(new_vs_indices[vt])

        self.gemm_vs_raw = new_gemm_vs
        self.gemm_vs = build_gemm_vs(new_gemm_vs, self, self.gemm_vs.shape[1])

        self.v_mask = self.v_mask[self.v_mask]
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()

    def cleanWithFace(self, edges_mask, face_mask, groups):
        edges_mask = edges_mask.astype(bool)
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]

        face_mask = face_mask.astype(bool)
        torch_mask = torch.from_numpy(face_mask.copy())
        self.gemm_faces = self.gemm_faces[face_mask]
        self.faces = self.faces[face_mask]
        self.face_areas = self.face_areas[face_mask]
        self.edges_in_face = self.edges_in_face[face_mask]

        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_edge_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_edge_indices[-1] = -1
        new_edge_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_edge_indices[self.gemm_edges[:, :]]
        self.edges_in_face[:, :] = new_edge_indices[self.edges_in_face[:, :]]

        face_mask = np.concatenate([face_mask, [False]])
        new_face_indices = np.zeros(face_mask.shape[0], dtype=np.int32)
        new_face_indices[-1] = -1
        new_face_indices[face_mask] = np.arange(0, np.ma.where(face_mask)[0].shape[0])
        self.gemm_faces[:, :] = new_face_indices[self.gemm_faces[:, :]]

        for v_index, ve in enumerate(self.ve):
            update_ve = []
            for e in ve:
                update_ve.append(new_edge_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve

        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()

    def build_faces(self):
        self.faces = []
        gemm = np.array(self.gemm_edges)
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32)
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0])
        for edge_index in range(len(gemm)):
            cycles = self.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                self.faces.append(self.__cycle_to_face(cycle, new_indices))
        self.faces = np.asarray(self.faces)

    def export(self, file=None, vcolor=None, extension='.obj'):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename)
                if (file_extension == ''):
                    file_extension = extension
                file = '%s/%s_%d%s' % (self.export_folder, filename, self.pool_count, file_extension)
            else:
                return
        faces = []
        vs = self.vs[self.v_mask]
        gemm = np.array(self.gemm_edges)
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32)
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0])
        for edge_index in range(len(gemm)):
            cycles = self.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                faces.append(self.__cycle_to_face(cycle, new_indices))
        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in self.edges:
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def export_raw(self, file=None, vcolor=None, extension='.obj'):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename)
                if (file_extension == ''):
                    file_extension = extension
                file = '%s/%s_%d%s' % (self.export_folder, filename, self.pool_count, file_extension)
            else:
                return

        with open(file, 'w+') as f:
            for vi, v in enumerate(self.vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(self.faces) - 1):
                f.write("f %d %d %d\n" % (
                self.faces[face_id][0] + 1, self.faces[face_id][1] + 1, self.faces[face_id][2] + 1))
            f.write("f %d %d %d" % (self.faces[-1][0] + 1, self.faces[-1][1] + 1, self.faces[-1][2] + 1))
            # for edge in self.edges:
            #     f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            edge_key = 0
            with os.fdopen(fh, 'w') as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == 'e':
                            new_file.write('%s %d' % (line.strip(), cur_segments[edge_key]))
                            if edge_key < len(cur_segments):
                                edge_key += 1
                                new_file.write('\n')
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['edges_mask']):
                cur_segments = segments[:len(self.history_data['edges_mask'][i])]
                cur_segments = cur_segments[self.history_data['edges_mask'][i]]

    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face

    def init_history(self):
        if self.opt.feat_from == 'edge':
            self.history_data = {
                'groups': [],
                'gemm_edges': [self.gemm_edges.copy()],
                'occurrences': [],
                'old2current': np.arange(self.edges_count, dtype=np.int32),
                'current2old': np.arange(self.edges_count, dtype=np.int32),
                'edges_mask': [torch.ones(self.edges_count, dtype=torch.bool)],
                'edges_count': [self.edges_count],
                'v_mask': [self.v_mask.copy()],
                'edges': [self.edges.copy()],
                'sides': [self.sides.copy()],
                'face_areas': [self.face_areas.copy()]
            }
            if self.export_folder:
                self.history_data['collapses'] = MeshUnion(self.edges_count)
        elif self.opt.feat_from == 'face':
            self.history_data = {
                'groups': [],
                'gemm_faces': [self.gemm_faces.copy()],
                'gemm_edges': [self.gemm_edges.copy()],
                'occurrences': [],
                'old2current': np.arange(self.face_count, dtype=np.int32),
                'current2old': np.arange(self.face_count, dtype=np.int32),
                'edges_mask': [torch.ones(self.face_count, dtype=torch.bool)],
                'edges_count': [self.edges_count],
                'face_count': [self.face_count],
                'v_mask': [self.v_mask.copy()],
                'edges': [self.edges.copy()],
                'sides': [self.sides.copy()],
                'face_areas': [self.face_areas.copy()]
            }
        elif self.opt.feat_from == 'point':
            self.history_data = {
                'groups': [],
                'gemm_faces': [self.gemm_faces.copy()],
                'gemm_edges': [self.gemm_edges.copy()],
                'occurrences': [],
                'old2current': np.arange(self.vs_count, dtype=np.int32),
                'current2old': np.arange(self.vs_count, dtype=np.int32),
                'edges_mask': [torch.ones(self.vs_count, dtype=torch.bool)],
                'edges_count': [self.edges_count],
                'face_count': [self.face_count],
                'vs_count': [self.vs_count],
                'vs': [self.vs.copy()],
                'v_mask': [self.v_mask.copy()],
                'edges': [self.edges.copy()],
                'sides': [self.sides.copy()],
                'face_areas': [self.face_areas.copy()]
            }
            if self.export_folder:
                self.history_data['collapses'] = MeshUnion(self.face_count)

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data['collapses'].union(self.history_data['current2old'][source],
                                                 self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['edges_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            if self.export_folder:
                self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()

    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            if self.export_folder != '':
                self.history_data['edges_mask'].append(self.history_data['edges_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)
            self.history_data['v_mask'].append(self.v_mask.copy())
            self.history_data['edges'].append(self.edges.copy())
            self.history_data['sides'].append(self.sides.copy())
            self.history_data['face_areas'].append(self.face_areas.copy())

            if self.opt.feat_from == 'edge':
                self.history_data['old2current'][mask] = np.arange(self.edges_count, dtype=np.int32)
                self.history_data['current2old'][0: self.edges_count] = np.ma.where(mask)[0]
            elif self.opt.feat_from == 'face':
                self.history_data['old2current'][mask] = np.arange(self.face_count, dtype=np.int32)
                self.history_data['current2old'][0: self.face_count] = np.ma.where(mask)[0]
                self.history_data['gemm_faces'].append(self.gemm_faces.copy())
                self.history_data['face_count'].append(self.face_count)

            elif self.opt.feat_from == 'point':
                self.history_data['old2current'][mask] = np.arange(self.vs_count, dtype=np.int32)
                self.history_data['current2old'][0: self.vs_count] = np.ma.where(mask)[0]
                self.history_data['gemm_faces'].append(self.gemm_faces.copy())
                self.history_data['face_count'].append(self.face_count)
                self.history_data['vs'].append(self.vs.copy())
                self.history_data['vs_count'].append(self.vs_count)




    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]
        self.history_data['v_mask'].pop()
        self.v_mask = self.history_data['v_mask'][-1]
        self.history_data['edges'].pop()
        self.edges = self.history_data['edges'][-1]
        self.history_data['sides'].pop()
        self.sides = self.history_data['sides'][-1]
        self.history_data['face_areas'].pop()
        self.face_areas = self.history_data['face_areas'][-1]
        if self.opt.feat_from == 'face' or self.opt.feat_from == 'point':
            self.history_data['gemm_faces'].pop()
            self.gemm_faces = self.history_data['gemm_faces'][-1]
            # self.history_data['faces'].pop()
            # self.faces = self.history_data['faces'][-1]
            self.history_data['face_count'].pop()
            self.face_count = self.history_data['face_count'][-1]

            if self.opt.feat_from == 'point':
                self.history_data['vs'].pop()
                self.vs = self.history_data['vs'][-1]
                self.history_data['vs_count'].pop()
                self.vs_count = self.history_data['vs_count'][-1]

    def get_edge_areas(self):
        return self.edge_areas
