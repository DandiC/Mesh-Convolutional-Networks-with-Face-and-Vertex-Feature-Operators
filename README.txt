
#########			INSTRUCTIONS TO RUN THE CODE		#########

INSTALATION:

The code has been tested in Python 3.6 with the following dependencies:
  - cython 0.27.3
  - numpy 1.15.0
  - pip 19.2.2
  - python 3.6.8
  - pytorch 1.3.0
  - tensorboardX(https://github.com/lanpa/tensorboardX) for training visualization (OPTIONAL).
  
Instructions to install the dependencies in Anaconda:
	conda env create -f environment.yml
	conda activate face_vertex_meshcnn
	
RUNNING THE CODE:

Executing the script train.py will train and test a face-based network in the SHREC16 dataset. Results will be saved under 'checkpoints/shrec_face'.

The folder 'scripts' contains bash scripts with the training configuration for classification and segmentation of different datasets. For instance, to train a vertex-based segmentation network on COSEG Aliens, one can execute the following command:
	bash ./scripts/coseg_seg/train_aliens_vertex.shrec_face
	
To test a previously trained model, one can run test.py with the same configuration as the trained model.

IMPORTANT NOTE: Once a network has been trained/tested in a specific dataset using a certain primitive (e.g., face-based), if one wants to train/test a new network with a different primitive (e.g., vertex-based), the flag --clean_data must be used to remove the cache files of the meshes. Otherwise the code will throw an error.

DISCLAIMER: The code included in this submission has not been completely cleaned. After acceptance, we will publicly release a more polished version of the code.
