MOOSS: Multi-Objective Optimization for Synthetic-to-Real Style Transfer

```
git clone --recurse-submodules git@github.com:xxxxx/MOOSS.git
conda create -n mooss python=3.12.11
conda activate mooss
cd MOOSS
# torch 2.8.0
# cuda 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install uv
uv pip install -r requirements.txt
uv pip install xformers
```


```
python run_pipeline.py
```