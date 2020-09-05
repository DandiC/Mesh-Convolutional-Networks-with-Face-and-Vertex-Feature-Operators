from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import DataLoader
import numpy as np
import os

if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = DataLoader(opt)

    sseg_face_folder = os.path.join(opt.dataroot,'sseg_face')
    seg_face_folder = os.path.join(opt.dataroot, 'seg_face')
    if not os.path.exists(sseg_face_folder):
        os.makedirs(sseg_face_folder)
    if not os.path.exists(seg_face_folder):
        os.makedirs(seg_face_folder)

    sseg_vt_folder = os.path.join(opt.dataroot, 'sseg_vt')
    seg_vt_folder = os.path.join(opt.dataroot, 'seg_vt')
    if not os.path.exists(sseg_vt_folder):
        os.makedirs(sseg_vt_folder)
    if not os.path.exists(seg_vt_folder):
        os.makedirs(seg_vt_folder)

    for i, data in enumerate(dataset):
        labels = data['label'] + 1
        soft_labels = data['soft_label']
        for m, mesh in enumerate(data['mesh']):
            mesh_edge_labels = labels[m]

            # Compute face labels
            mesh_face_labels = []
            mesh_face_soft_labels = np.zeros((mesh.face_count, soft_labels.shape[-1]))
            for f, _ in enumerate(mesh.faces):
                face_labels = mesh_edge_labels[mesh.edges_in_face[f]]
                assert 0 not in face_labels
                uniques, counts = np.unique(face_labels,return_counts=True)
                mesh_face_labels.append(uniques[np.argmax(counts)].astype(int))
                for u, unique in enumerate(uniques):
                    mesh_face_soft_labels[f, int(unique)-1] = counts[u]/len(face_labels)

            # Save face labels
            np.savetxt(os.path.join(sseg_face_folder, mesh.filename.replace('.obj', '.seseg')), mesh_face_soft_labels,
                       delimiter=' ', fmt='%.6f')
            np.savetxt(os.path.join(seg_face_folder, mesh.filename.replace('.obj', '.eseg')),
                       np.asarray(mesh_face_labels), fmt='%i')

            # Compute vertex labels
            mesh_vs_labels = []
            mesh_vs_soft_labels = np.zeros((mesh.vs_count, soft_labels.shape[-1]))
            for v, _ in enumerate(mesh.vs):
                vs_labels = mesh_edge_labels[mesh.ve[v]]
                assert 0 not in vs_labels
                uniques, counts = np.unique(vs_labels, return_counts=True)
                mesh_vs_labels.append(uniques[np.argmax(counts)].astype(int))
                for u, unique in enumerate(uniques):
                    mesh_vs_soft_labels[v, int(unique)-1] = counts[u] / len(vs_labels)

            # Save vertex labels
            np.savetxt(os.path.join(sseg_vt_folder, mesh.filename.replace('.obj', '.seseg')),
                       mesh_vs_soft_labels,
                       delimiter=' ', fmt='%.6f')
            np.savetxt(os.path.join(seg_vt_folder, mesh.filename.replace('.obj', '.eseg')),
                       np.asarray(mesh_vs_labels), fmt='%i')




