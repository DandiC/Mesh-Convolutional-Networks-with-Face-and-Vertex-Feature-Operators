import numpy as np
import os
import ntpath
import random
import math

def fill_mesh(mesh2fill, file: str, opt, faces=None,vertices=None, feat_from='face'):
    if file==None:
        mesh_data = from_faces_and_vertices(faces,vertices, vertex_features=opt.vertex_features)
    else:
        load_path = get_mesh_path(file, opt.num_aug)
        if os.path.exists(load_path):
            mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True)
        else:
            mesh_data = from_scratch(file, opt)
            np.savez_compressed(load_path, gemm_edges=mesh_data.gemm_edges, vs=mesh_data.vs, edges=mesh_data.edges,
                                edges_count=mesh_data.edges_count, ve=mesh_data.ve, v_mask=mesh_data.v_mask,
                                filename=mesh_data.filename, sides=mesh_data.sides, edge_lengths=mesh_data.edge_lengths,
                                edge_areas=mesh_data.edge_areas, edge_features=mesh_data.edge_features,
                                face_features=mesh_data.face_features, faces=mesh_data.faces,
                                face_areas=mesh_data.face_areas, gemm_faces=mesh_data.gemm_faces,
                                face_count=mesh_data.face_count, edges_in_face=mesh_data.edges_in_face, ef=mesh_data.ef,
                                gemm_vs=mesh_data.gemm_vs, gemm_vs_raw=mesh_data.gemm_vs_raw,
                                vs_count=mesh_data.vs_count, vf=mesh_data.vf, vs_normals=mesh_data.vs_normals,
                                vertex_features=mesh_data.vertex_features)

    mesh2fill.vs = mesh_data['vs']
    mesh2fill.vs_count = int(mesh_data['vs_count'])
    mesh2fill.gemm_vs = mesh_data['gemm_vs']
    mesh2fill.gemm_vs_raw = mesh_data['gemm_vs_raw']
    mesh2fill.edges = mesh_data['edges']
    mesh2fill.gemm_edges = mesh_data['gemm_edges']
    mesh2fill.edges_count = int(mesh_data['edges_count'])
    mesh2fill.ve = mesh_data['ve']
    mesh2fill.v_mask = mesh_data['v_mask']
    mesh2fill.filename = str(mesh_data['filename'])
    mesh2fill.edge_lengths = mesh_data['edge_lengths']
    mesh2fill.edge_areas = mesh_data['edge_areas']
    mesh2fill.sides = mesh_data['sides']
    mesh2fill.faces = mesh_data['faces']
    mesh2fill.face_areas = mesh_data['face_areas']
    mesh2fill.gemm_faces = mesh_data['gemm_faces']
    mesh2fill.face_count = int(mesh_data['face_count'])
    mesh2fill.edges_in_face = mesh_data['edges_in_face']
    mesh2fill.ef = mesh_data['ef']
    mesh2fill.vf = mesh_data['vf']
    mesh2fill.vs_normals = mesh_data['vs_normals']

    if feat_from == 'edge':
        mesh2fill.features = mesh_data['edge_features']
    elif feat_from == 'face':
        mesh2fill.features = mesh_data['face_features']
    elif feat_from == 'point':
        mesh2fill.features = mesh_data['vertex_features']
    else:
        raise ValueError(opt.feat_from, 'Wrong parameter value in --feat_from')

def get_mesh_path(file: str, num_aug: int):
    filename, _ = os.path.splitext(file)
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    load_dir = os.path.join(dir_name, 'cache')
    load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, num_aug)))
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    return load_file

def from_scratch(file, opt):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.vf = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.vs, faces = fill_from_file(mesh_data, file)
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    mesh_data.faces, mesh_data.face_areas, mesh_data.face_normals = remove_non_manifolds(mesh_data, faces)
    mesh_data.face_count = mesh_data.faces.shape[0]
    mesh_data.vs_count = mesh_data.vs.shape[0]
    if opt.num_aug > 1:
        mesh_data.faces = augmentation(mesh_data, opt, mesh_data.faces)
    build_gemm(mesh_data, n_neighbors=opt.n_neighbors)
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
    mesh_data.edge_features, mesh_data.face_features, mesh_data.vertex_features = extract_features(mesh_data, vf=opt.vertex_features)
    return mesh_data

def from_faces_and_vertices(faces,vertices, vertex_features=None):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = vertices
    mesh_data.faces=faces
    mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    mesh_data.faces, mesh_data.face_areas, mesh_data.face_normals = remove_non_manifolds(mesh_data, faces)
    # mesh_data.face_normals, mesh_data.face_areas = compute_face_normals_and_areas(mesh_data, faces)
    mesh_data.face_count = mesh_data.faces.shape[0]
    mesh_data.vs_count = mesh_data.vs.shape[0]

    build_gemm(mesh_data)

    mesh_data.edge_features, mesh_data.face_features, mesh_data.vertex_features = extract_features(mesh_data, vf=vertex_features)

    return mesh_data
# Fills vertices and faces by reading the OBJ file line by line
def fill_from_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)

    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def remove_non_manifolds(mesh, faces):
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    face_normals, face_areas = compute_face_normals_and_areas(mesh, faces)
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask], face_normals[mask]


def build_gemm(mesh, n_neighbors=6):
    mesh.ve = [[] for _ in mesh.vs]
    mesh.vf = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    faces_edges = []
    face_nb = -np.ones(mesh.faces.shape)
    face_nb_count = []
    ef = []
    edges_in_faces = -np.ones(mesh.faces.shape)
    point_nb = [set() for _ in range(mesh.vs.shape[0])]
    for face_id, face in enumerate(mesh.faces):
        face_edges = []
        face_nb_count.append(0)
        for i in range(3):
            mesh.vf[face[i]].append(face_id)
            cur_edge = (face[i], face[(i + 1) % 3])
            face_edges.append(cur_edge)
            point_nb[face[i]].add(face[(i + 1) % 3])
            point_nb[face[i]].add(face[(i + 2) % 3])
        for idx, edge in enumerate(face_edges):
            edge = tuple(sorted(list(edge)))
            face_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                ef.append(list([-1, -1]))
                ef[edges_count][0]=face_id
                edges_count += 1
            else:
                index = edge2key[edge]
                ef[index][1]=face_id
            mesh.edge_areas[edge2key[edge]] += mesh.face_areas[face_id] / 3

            #Find face neighbors
            neighbor = np.where(edges_in_faces==edge2key[edge])[0]
            if neighbor.size == 1:
                n_id = neighbor[0]
                face_nb[n_id, face_nb_count[n_id]] = face_id
                face_nb[face_id, face_nb_count[face_id]] = n_id
                face_nb_count[n_id] += 1
                face_nb_count[face_id] += 1
            #Set edge indices in the face
            edges_in_faces[face_id,idx] = edge2key[edge]

        for idx, edge in enumerate(face_edges):
            #Edge neighbors
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[face_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[face_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2

        for idx, edge in enumerate(face_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[face_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[face_edges[(idx + 2) % 3]]] - 2
        faces_edges.append(face_edges)

    compute_vs_normals(mesh)
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_vs_raw = point_nb
    mesh.gemm_vs = build_gemm_vs(point_nb, mesh, n_neighbors)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.gemm_faces = face_nb.astype(np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(mesh.face_areas) #todo whats the difference between edge_areas and edge_lenghts?
    mesh.edges_in_face = edges_in_faces.astype(np.int64)
    mesh.ef = np.array(ef, dtype=np.int64)


def build_gemm_vs(point_nb, mesh, n_neighbors):
    gemm_vs = -np.ones((mesh.vs.shape[0], n_neighbors), dtype=int)

    for i, gemm in enumerate(point_nb):
        l_gemm = list(gemm)

        dist = np.linalg.norm(mesh.vs[l_gemm] - mesh.vs[i], axis=1)
        order = np.argsort(dist)
        gemm_vs[i, :min(n_neighbors, len(gemm))] = np.asarray(l_gemm)[order[:min(n_neighbors, len(gemm))]]

    return gemm_vs

def compute_vs_normals(mesh):
    mesh.vs_normals = np.zeros(mesh.vs.shape)
    for v_id, vertices in enumerate(mesh.vf):
        mesh.vs_normals[v_id] = np.sum(mesh.face_normals[vertices,:], axis=0)
        if not np.all(mesh.vs_normals[v_id] == 0):
            mesh.vs_normals[v_id]=mesh.vs_normals[v_id]/np.linalg.norm(mesh.vs_normals[v_id])

def compute_face_normals_and_areas(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    if np.any(face_areas[:, np.newaxis] == 0):
        print('has zero area face: %s' % mesh.filename)
        face_areas[face_areas == 0] = 0.000001
    face_normals /= face_areas[:, np.newaxis]
    face_areas *= 0.5
    return face_normals, face_areas


# Data augmentation methods
def augmentation(mesh, opt, faces=None):
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        faces = flip_edges(mesh, opt.flip_edges, faces)
    return faces


def post_augmentation(mesh, opt):
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)


def slide_verts(mesh, prct):
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(mesh, edge_points).squeeze() #todo make fixed_division epsilon=0
    thr = np.mean(dihedral) + np.std(dihedral)
    vids = np.random.permutation(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]
            if min(dihedral[edges]) > 2.65:
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) * (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted += 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)


def scale_verts(mesh, mean=1, var=0.1):
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)


def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count)
    target = int(prct * edge_count)
    flipped = 0
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]]))))
            if new_edge in edges_dict:
                continue
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    return faces


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face

def check_area(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    mesh.edge_lengths = edge_lengths


def extract_features(mesh, vf=None):
   #Extraction of Edge Features
    edge_features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                feature = extractor(mesh, edge_points)
                edge_features.append(feature)
            edge_features = np.concatenate(edge_features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad edge features')
    # Extraction of Face Features
    face_features = []
    with np.errstate(divide='raise'):
        try:
            for extractor in [face_angles, face_dihedral_angles, area_ratios]:
                feature = extractor(mesh)
                face_features.append(feature)
            face_features = np.concatenate(face_features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad face features')

    # Extraction of Vertex Features
    vertex_features = []
    with np.errstate(divide='raise'):
        try:
            for f in vf:
                if f == 'coord':
                    feature = vertex_coordinates(mesh)
                elif f == 'norm':
                    feature = vertex_normals(mesh)
                elif f == 'mean_c':
                    feature = mean_curvature(mesh, edge_features)
                elif f == 'gaussian_c':
                    feature = gaussian_curvature(mesh)
                else:
                    raise ValueError(vertex_features, 'Wrong value value in --vertex_features')
                vertex_features.append(feature)
            vertex_features = np.concatenate(vertex_features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad vertex features')

    return edge_features, face_features, vertex_features

def gaussian_curvature(mesh):
    gaussian_curv = np.zeros((1,mesh.vs.shape[0]))
    for v_i, vt in enumerate(mesh.vs):
        Ai = np.sum(mesh.face_areas[mesh.vf[v_i]]) / 3
        if Ai == 0:
            # To avoid dividing by 0.
            gaussian_curv[0, v_i] = 0
        else:
            for j, f_j in enumerate(mesh.vf[v_i]):
                neighbors = mesh.faces[f_j,mesh.faces[f_j]!=v_i]
                edge_a = mesh.vs[neighbors[0]] - vt
                edge_b = mesh.vs[neighbors[1]] - vt

                edge_a /= fixed_division(np.linalg.norm(edge_a, ord=2), epsilon=0.0001)
                edge_b /= fixed_division(np.linalg.norm(edge_b, ord=2), epsilon=0.0001)
                dot = np.sum(edge_a * edge_b).clip(-1, 1)
                gaussian_curv[0,v_i] += np.arccos(dot)
            gaussian_curv[0,v_i] = (2*np.pi - gaussian_curv[0,v_i])/Ai
    return gaussian_curv

def get_cotangent_laplacian_beltrami(mesh, edge_features):
    laplacian = np.zeros(mesh.vs.shape)
    for v_i, vt in enumerate(mesh.vs):
        Ai=np.sum(mesh.face_areas[mesh.vf[v_i]])/3
        if Ai == 0:
            laplacian[v_i,:] = 0
        else:
            for j, v_j in enumerate(mesh.gemm_vs_raw[v_i]):
                #Get edge between two vertices
                # TODO: Optimize this
                edge_id = np.argmax(np.logical_or(np.logical_and(mesh.edges[:, 0] == v_i, mesh.edges[:, 1] == v_j),
                                                  np.logical_and(mesh.edges[:, 0] == v_j, mesh.edges[:, 1] == v_i)))
                angles = edge_features[1:3,edge_id]
                laplacian[v_i,:] += (1/math.tan(angles[0])+1/math.tan(angles[1]))*(mesh.vs[v_j] - vt)
            laplacian[v_i,:] = laplacian[v_i]/(2*Ai)
    return laplacian

def mean_curvature(mesh, edge_features):
    laplacians = get_cotangent_laplacian_beltrami(mesh, edge_features)
    return np.expand_dims(np.linalg.norm(laplacians, axis=1)/2, 0)

def vertex_normals(mesh):
    return np.transpose(mesh.vs_normals)

def vertex_coordinates(mesh):
    return np.transpose(mesh.vs)

def face_angles(mesh):
    angles_a = get_angles(mesh, 0)
    angles_b = get_angles(mesh, 1)
    angles_c = get_angles(mesh, 2)
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0), np.expand_dims(angles_c, 0)), axis=0)
    return np.sort(angles, axis=0)

def get_angles(mesh, side):
    edges_a = mesh.vs[mesh.faces[:, (side-1) % 3]] - mesh.vs[mesh.faces[:, side]]
    edges_b = mesh.vs[mesh.faces[:, (side+1) % 3]] - mesh.vs[mesh.faces[:, side]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)

def face_dihedral_angles(mesh):

    #Get normals from neighbors
    normals_a = mesh.face_normals[mesh.gemm_faces[:,0]]
    normals_b = mesh.face_normals[mesh.gemm_faces[:, 1]]
    normals_c = mesh.face_normals[mesh.gemm_faces[:, 2]]

    #Dot product between normals
    dot_a = np.sum(normals_a * mesh.face_normals, axis=1).clip(-1, 1)
    dot_b = np.sum(normals_b * mesh.face_normals, axis=1).clip(-1, 1)
    dot_c = np.sum(normals_c * mesh.face_normals, axis=1).clip(-1, 1)

    #Dihedral angle between two faces is 180-arccos(dot product)
    angles_a = np.pi - np.arccos(dot_a)
    angles_b = np.pi - np.arccos(dot_b)
    angles_c = np.pi - np.arccos(dot_c)

    #Mask if neighbor does not exist
    mask = mesh.gemm_faces == -1
    angles_a[mask[:,0]] = 0
    angles_b[mask[:,1]] = 0
    angles_c[mask[:,2]] = 0

    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0), np.expand_dims(angles_c, 0)), axis=0)

    return np.sort(angles, axis=0)

def area_ratios(mesh):

    #Get areas from neighbors
    areas_a = mesh.face_areas[mesh.gemm_faces[:,0]]
    areas_b = mesh.face_areas[mesh.gemm_faces[:, 1]]
    areas_c = mesh.face_areas[mesh.gemm_faces[:, 2]]

    # Mask if neighbor does not exist
    mask = mesh.gemm_faces == -1
    areas_a[mask[:, 0]] = 0
    areas_b[mask[:, 1]] = 0
    areas_c[mask[:, 2]] = 0

    #compute ratios
    ratios = np.concatenate((np.expand_dims(areas_a / mesh.face_areas, 0),
                             np.expand_dims(areas_b / mesh.face_areas, 0),
                             np.expand_dims(areas_c / mesh.face_areas, 0)), axis=0)

    return np.sort(ratios, axis=0)

def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def get_edge_points(mesh):
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
    return edge_points


def get_side_points(mesh, edge_id):
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals

def get_opposite_angles(mesh, edge_points, side):
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                   ord=2, axis=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths


def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.01
    else:
        to_div += epsilon
    return to_div
