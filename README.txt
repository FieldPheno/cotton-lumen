conda create -n cotton python=3.6

conda activate cotton

pip install tensorflow-gpu==1.4.0
pip install h5py==2.10.0
#optional version decrease to avoid warnings
pip install scikit-image==0.16.2
pip install keras==2.0.8
#optional numpy version to avoid warning
pip install numpy==1.16.4
pip install imgaug
pip install scikit-learn
pip install git+https://github.com/keras-team/keras-preprocessing.git

git clone https://github.com/matterport/Mask_RCNN.git

cd Mask_RCNN

pip install -r requirements.txt

python setup.py install

wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5



#running conda env in jupyter
#activate env
conda install ipykernel
ipython kernel install --user --name=cotton


conda install conda-forge cudatoolkit=8.0 cudnn=6.0.0