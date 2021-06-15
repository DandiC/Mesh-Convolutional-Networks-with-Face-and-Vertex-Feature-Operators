import glob
import os
import subprocess

if __name__ == '__main__':
    cmd = '"C:\Program Files\VCG\MeshLab\meshlabserver.exe" -i {} -o {}'

    input_folder = 'datasets/badModelnet'
    input_format = '.off'

    output_folder = 'datasets/badModelnet'
    output_format = '.obj'

    mesh_files = glob.glob(os.path.join(input_folder, '*' + input_format))

    for file in mesh_files:
        export_name = file.replace(input_folder, output_folder).replace(input_format, output_format)
        folder = os.path.dirname(export_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        command = cmd.format(file, export_name)
        print(command)
        env = os.environ
        subprocess.call(command, shell=True, env=env)