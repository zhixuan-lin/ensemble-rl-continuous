# DrQv2

## Mila cluster

Virtual environment

```
conda create -n drqv2 python=3.9
conda activate drqv2

```

Other requirements

```
pip install -r requirements.txt
```

PyTorch. We don't put this in `requirements.txt` because the `--index-url` option is needed. And if we put it in `requirements.txt` it will be applied to all dependencies by default. See [this](https://stackoverflow.com/questions/2477117/pip-requirements-txt-with-alternative-index).

```
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
```



## Compute canada 

> Run this on login nodes 
> ```
> module load cuda/11.4
> module load cudnn/8.2
> ```

create virtual env

```
module load python/3.9
mkdir -p ~/.venv
cd ~/.venv
virtualenv scaling
cd -
source ~/.venv/scaling/bin/activate
```

Install JAX and flax

```
pip install jaxlib==0.1.69 --no-index
pip install jax==0.2.25 --no-index
pip install flax==0.3.5
```


Install mujoco 2.3.5 (for dm_control 1.0.12)

```
mkdir -p $HOME/scratch/tmp/
cd $HOME/scratch/tmp/
wget https://files.pythonhosted.org/packages/02/c7/bec8fce4bbe70e11d6e81f78f9c6413eaaa02952db2be9d5a1a0fd8f8c0d/mujoco-2.3.5-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
mv mujoco-2.3.5-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl mujoco-2.3.5-cp39-none-linux_x86_64.whl
pip install mujoco-2.3.5-cp39-none-linux_x86_64.whl
cd -
```


Other requirements.

```
pip install -r requirements.txt
```

NO idea why but this works

```
pip install pyopengl==3.1.6 --no-index
pip install numpy==1.22.2 --no-index
```

