
```shell

# install dependencies
pip install -r requirements.txt

# MLM
python dynamic-masking.py
python static-masking.py

# NSP
python nsp.py

# analyse 
python analyse.py

# replace LayerNorm with RMSNorm
python -m rmsnorm.train_rmsnorm


```