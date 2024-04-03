conda create -n caramlicl python=3.9 -y
conda activate caramlicl
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl -y
pip install -e . --user
