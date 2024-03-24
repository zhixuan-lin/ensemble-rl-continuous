# Ensemble RL

This is code for the MuJoCo experiments in the following ICLR 24 paper:

> [The Curse of Diversity in Ensemble-Based Exploration](https://openreview.net/forum?id=M3QXCOTTk4)
>
> *Zhixuan Lin, Pierluca D'Oro, Evgenii Nikishin, Aaron Courville*

The codebase is built upon [jaxrl](https://github.com/ikostrikov/jaxrl).

## Dependencies

Create `conda` environments and activate

```
conda create -n ensemble-rl-continuous python=3.9
conda activate ensemble-rl-continuous
```

Install Jax (GPU) and flax. Note this requires CUDA 11.8:

```
pip install "jax[cuda11_pip]==0.4.24" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.8.0
```

Install other requirements:

```
pip install -r requirements.txt
```

For some reason `gym==0.25.2`'s vectorized env does not work with high versions of numpy. So downgrading:

```
pip install numpy==1.23.5
```



## Running Experiments

Login to your wandb account with `wandb login`.

The default command for Ensemble SAC (running 10 seeds in parallel) with replay buffer size 200k is:

```bash
python train_cel_parallel.py \
    --exp debug \
    --save_dir ./output \
    --config examples/configs/cel_default.py \
    --env_name HalfCheetah-v4 \
    --config.n 10 \
    --config.replay_buffer_size 200000 \
    --seed 0,1,2,3,4,5,6,7,8,9 \

```

Results are saved to `./output`. Wandb visualization is also available under the project name `ensemble-rl-continuous`.

The configurations for the experiments in the main paper are as follows:

* SAC with replay buffer size `B`, with `B=200000` as an example (Figure 7):

  ```bash
  B=200000
  python train_cel_parallel.py \
      --exp debug \
      --save_dir ./output \
      --config examples/configs/cel_default.py \
      --env_name HalfCheetah-v4 \
      --config.n 1 \
      --config.replay_buffer_size ${B} \
      --seed 0,1,2,3,4,5,6,7,8,9 \
  ```

* Ensemble SAC with replay buffer sizes `B`, with `B=200000` as an example (Figure 7):

  ```bash
  B=200000
  python train_cel_parallel.py \
      --exp debug \
      --save_dir ./output \
      --config examples/configs/cel_default.py \
      --env_name HalfCheetah-v4 \
      --config.n 10 \
      --config.replay_buffer_size ${B} \
      --seed 0,1,2,3,4,5,6,7,8,9 \
  ```

* CERL with replay buffer size `B`, with `B=200000` as an example (Figure 7):

  ```bash
  B=200000
  python train_cel_parallel.py \
      --exp debug \
      --save_dir ./output \
      --config examples/configs/cel_default.py \
      --env_name HalfCheetah-v4 \
      --config.n 10 \
      --config.replay_buffer_size ${B} \
      --config.cel \
      --config.aux_huber \
      --config.huber_delta 10 \
      --seed 0,1,2,3,4,5,6,7,8,9 \
  ```

* $90$%-tandem (Figure 8, Appendix)

  ```bash
  python train_cel_parallel.py \
      --exp debug \
      --save_dir ./output \
      --config examples/configs/cel_default.py \
      --env_name HalfCheetah-v4 \
      --config.n 2 \
      --config.tandem \
      --config.active_prob 0.9 \
      --config.replay_buffer_size 200000 \
      --seed 0,1,2,3,4,5,6,7,8,9 \
  ```

# Citation

If you find this code useful, please cite the following:

```
@inproceedings{
lin2024the,
title={The Curse of Diversity in Ensemble-Based Exploration},
author={Zhixuan Lin and Pierluca D'Oro and Evgenii Nikishin and Aaron Courville},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=M3QXCOTTk4}
}
```

And the jaxrl repo:

```
@misc{jaxrl,
  author = {Kostrikov, Ilya},
  doi = {10.5281/zenodo.5535154},
  month = {10},
  title = {{JAXRL: Implementations of Reinforcement Learning algorithms in JAX}},
  url = {https://github.com/ikostrikov/jaxrl},
  year = {2021}
}
```

