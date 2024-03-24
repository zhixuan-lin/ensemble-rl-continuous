import os
import sys
import random
import wandb
from datetime import datetime
import uuid
import shutil
# os.environ["WANDB_MODE"] = "offline"

import numpy as np
from numpy.random import default_rng
import tqdm
from absl import app, flags
from ml_collections import config_flags
#from tensorboardX import SummaryWriter

from jaxrl.agents import CELLearner
from jaxrl.datasets import ReplayBuffer, ParallelReplayBuffer
from jaxrl.datasets.replay_buffer_multi import ReplayBufferMulti
from jaxrl.evaluation import evaluate, evaluate_parallel, evaluate_parallel_and_record, caculate_estimation_bias, evaluate_parallel_seed, evaluate_parallel_seed_cel
from jaxrl.utils import make_env
import json
import gym
import jax.numpy as jnp
from jaxrl.networks.common import save_model, load_model

from collections import deque, OrderedDict
from typing import Optional, List
from pathlib import Path
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', 'debug', 'Experiment description (not actually used).')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_list('seed', [42], 'Random seed.')
flags.DEFINE_integer('eval_episodes', 30,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_boolean('eval_value_bias', True, 'Whether to evaluate value prediction bias')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_integer('reset_interval', int(1e5), 'Periodicity of resets.')
flags.DEFINE_boolean('resets', False, 'Periodically reset last actor / critic layers.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('save_1M_buffer', False, 'Save buffer after 1M steps.')
flags.DEFINE_boolean('save_buffer', False, 'Save replay buffer')
flags.DEFINE_boolean('save_agent', False, 'Save replay buffer')
flags.DEFINE_boolean('save_checkpoint', True, 'Save checkpoint')
flags.DEFINE_integer('checkpoint_interval', 250000, 'Save checkpoint')
flags.DEFINE_boolean('resume_checkpoint', True, 'Save checkpoint')
flags.DEFINE_integer('nsteps', 1, 'N for n-step returns.')
# wandb tags
flags.DEFINE_list('tags', None, 'wandb tags')
flags.DEFINE_string('notes', None, 'wandb notes')
config_flags.DEFINE_config_file(
    'config',
    # 'examples/configs/sac_default.py',
    'examples/configs/daq_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

def log_multiple_seeds_to_wandb(step, infos):
    dict_to_log = {}
    for info_key in infos:
        for seed_id, value in enumerate(infos[info_key]):
            dict_to_log[f'seed_{FLAGS.seed[seed_id]}/{info_key}'] = value
    wandb.log(dict_to_log, step=step)

def load_latest_checkpoint(save_dir: str, seeds: List, buffer: ParallelReplayBuffer, agent: CELLearner, stats: np.ndarray, eval_returns: np.ndarray):
    save_dir = Path(save_dir)
    sentinel_list = list(save_dir.glob('sentinel_*'))
    if len(sentinel_list) == 0:
        return None
    def get_step(path):
        step = int(path.name.split('_')[-1])
        return step

    latest_step = max(map(get_step, sentinel_list))

    checkpoint_dir = Path(save_dir) / f'checkpoint_{latest_step}'

    assert checkpoint_dir.is_dir()

    # Save stuff
    with (checkpoint_dir / 'seeds.pickle').open('rb') as f:
        loaded_seeds = pickle.load(f)
        assert list(seeds) == list(loaded_seeds)
    with (checkpoint_dir / 'stats.pickle').open('rb') as f:
        loaded_stats = pickle.load(f)
        for i in range(len(seeds)):
            stats[i] = loaded_stats[i]

    with (checkpoint_dir / 'eval_returns.pickle').open('rb') as f:
        loaded_eval_returns = pickle.load(f)
        for i in range(len(seeds)):
            eval_returns[i] = loaded_eval_returns[i]


    buffer.load(checkpoint_dir)
    agent.actor = load_model(str(checkpoint_dir / 'actor'), agent.actor)
    agent.critic = load_model(str(checkpoint_dir / 'critic'), agent.critic)
    agent.target_critic = load_model(str(checkpoint_dir / 'target_critic'), agent.target_critic)
    agent.temp = load_model(str(checkpoint_dir / 'temp'), agent.temp)
    return latest_step


def save_checkpoint(save_dir: str, seeds: List, completed_steps: int, buffer: ParallelReplayBuffer, agent: CELLearner, stats: np.ndarray, eval_returns: np.ndarray, to_remove: Optional[int] = None):
    checkpoint_dir = Path(save_dir) / f'checkpoint_{completed_steps}'
    sentinel_file = Path(save_dir) / f'sentinel_{completed_steps}'
    if checkpoint_dir.exists():
        # Overwrite
        shutil.rmtree(str(checkpoint_dir))
        sentinel_file.unlink(missing_ok=True)

    checkpoint_dir.mkdir()

    # Save stuff
    with (checkpoint_dir / 'seeds.pickle').open('wb') as f:
        pickle.dump(list(seeds), f)
    with (checkpoint_dir / 'stats.pickle').open('wb') as f:
        pickle.dump(stats, f)
    with (checkpoint_dir / 'eval_returns.pickle').open('wb') as f:
        pickle.dump(eval_returns, f)

    buffer.save(checkpoint_dir)
    save_model(str(checkpoint_dir / 'actor'), agent.actor)
    save_model(str(checkpoint_dir / 'critic'), agent.critic)
    save_model(str(checkpoint_dir / 'target_critic'), agent.target_critic)
    save_model(str(checkpoint_dir / 'temp'), agent.temp)

    # All good, create sentinel
    with sentinel_file.open('w'):
        pass

    # Remove previous files
    if to_remove is not None:
        dir_to_remove = Path(save_dir) / f'checkpoint_{to_remove}'
        sentinel_to_remove = Path(save_dir) / f'sentinel_{to_remove}'
        sentinel_to_remove.unlink(missing_ok=True)
        if dir_to_remove.exists():
            shutil.rmtree(str(dir_to_remove))


def main(_):

    # suppress warning
    import logging
    logger = logging.getLogger()
    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return "check_types" not in record.getMessage()

    logger.addFilter(CheckTypesFilter())

    os.makedirs(FLAGS.save_dir, exist_ok=True)
#     summary_writer = SummaryWriter(
#         os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))


    FLAGS.seed = [int(s) for s in FLAGS.seed]
    print(FLAGS.seed)
    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None


    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))

    kwargs = dict(FLAGS.config)
    kwargs['nsteps'] = FLAGS.nsteps
    hidden_size = kwargs.pop('hidden_size')
    kwargs['hidden_dims'] = (hidden_size, hidden_size)
    algo = kwargs.pop('algo')
    updates_per_step = kwargs.pop('updates_per_step')
    replay_buffer_size = kwargs.pop('replay_buffer_size')

    num_seeds = len(FLAGS.seed)
    np.random.seed(FLAGS.seed[0])
    random.seed(FLAGS.seed[0])
    # rng = default_rng(FLAGS.seed)

    def create_env_fn(env_name, seed, video_folder):
        return lambda: make_env(env_name, seed, video_folder)

    env = gym.vector.SyncVectorEnv(
        [create_env_fn(FLAGS.env_name, int(seed), video_train_folder) for seed in FLAGS.seed])
    eval_env = gym.vector.SyncVectorEnv(
        [create_env_fn(FLAGS.env_name, int(seed), video_eval_folder) for seed in FLAGS.seed])


    id_path = Path(FLAGS.save_dir) / 'wandb_run_id.txt'
    if FLAGS.resume_checkpoint and id_path.is_file():
        with id_path.open('r') as f:
            run_id = f.read().strip()
            if run_id:
                logging.info(f'Resuming wandb run {run_id}.')
            else:
                run_id = None
    else:
        run_id = None

    wandb.init(
        project="ensemble-rl-continuous",
        config=all_kwargs,#{
        #     "env_name": FLAGS.env_name,
        #     "max_steps": FLAGS.max_steps,
        #     "resets": FLAGS.resets,
        #     "reset_interval": FLAGS.reset_interval,
        #     "replay_buffer_size": replay_buffer_size,
        #     "seed": FLAGS.seed,
        # }
        # TODO: 
        # dir=FLAGS.save_dir
        tags=FLAGS.tags,
        notes=FLAGS.notes,
        resume='allow',
        id=run_id
    )
    with id_path.open('w') as f:
        f.write(wandb.run.id)

    # Save config
    config_path = os.path.join(FLAGS.save_dir, 'config.json')
    if os.path.isfile(config_path):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_config_path = os.path.join(FLAGS.save_dir, f'config_{date_str}.json')
        os.rename(config_path, archive_config_path)
    with open(config_path, 'w') as f:
        json.dump(all_kwargs, f, indent=2)


    assert algo in ['cel']
    if algo == 'cel':
        agent = CELLearner(FLAGS.seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()

    action_dim = env.single_action_space.shape[0]
    replay_buffer = ParallelReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps, num_buffers=len(FLAGS.seed))
    # replay_buffer = ReplayBufferMulti(env.single_observation_space, action_dim,
                                 # replay_buffer_size or FLAGS.max_steps, n=len(FLAGS.seed))

    eval_returns = np.empty(num_seeds, dtype=list)
    stats = np.empty(num_seeds, dtype=list)
    for seed_id in range(num_seeds):
        stats[seed_id] = []
        eval_returns[seed_id] = []

    start_step = 1
    if FLAGS.resume_checkpoint:
        loaded_step = load_latest_checkpoint(FLAGS.save_dir, FLAGS.seed, replay_buffer, agent, stats, eval_returns)
        if loaded_step is not None:
            start_step = loaded_step + 1
            print(f'Loaded checkpoint from step {loaded_step}.')
    observation, done, reward, info = env.reset(), np.zeros(num_seeds, dtype=bool), np.zeros(num_seeds), {}
    episode_returns = np.zeros(num_seeds)
    batch = None
    agent.begin_episode(mask=np.ones(num_seeds, dtype=bool), eval_mode=False)
    for i in tqdm.tqdm(range(start_step, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm,
                       initial=start_step - 1,
                       total=FLAGS.max_steps):
        step_stats = np.empty(num_seeds, dtype=OrderedDict)
        for seed_id in range(num_seeds):
            step_stats[seed_id] = OrderedDict([('step', i)])

        if hasattr(agent, 'update_stats'):
            agent.update_stats({'total_steps': FLAGS.max_steps + 1, 'current_step': i, 'updates_per_step': updates_per_step, 'start_training': FLAGS.start_training})

        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            if algo == 'cel':
                action, _ = agent.sample_actions(observation, eval_mode=False)
            else:
                action = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)
        episode_returns += reward


        # if not done or ('TimeLimit.truncated' in info and info['TimeLimit.truncated']):
            # mask = 1.0
        # else:
            # mask = 0.0

        real_next_observation = []
        masks = []
        # Stupid stuff due to gym's stupid autoreset design
        for seed_id in range(num_seeds):
            if not done[seed_id] or ('TimeLimit.truncated' in info and info['TimeLimit.truncated'][seed_id]):
                masks.append(1.0)
            else:
                masks.append(0.0)
            if done[seed_id]:
                assert info['_final_observation'][seed_id]
                real_next_observation.append(info['final_observation'][seed_id])
            else:
                real_next_observation.append(next_observation[seed_id])
        mask = np.stack(masks, axis=0).astype(np.float32)
        real_next_observation = np.stack(real_next_observation, axis=0).astype(np.float32)

        replay_buffer.insert(observation, action, reward, mask, done.astype(np.float32),
                            real_next_observation)
        observation = next_observation


        # Begin episode for agent. For DMC, episode length always the same
        if done.any():
            # assert done.all()
            agent.begin_episode(mask=done, eval_mode=False)

        for seed_id in range(num_seeds):

            if done[seed_id]:
                wandb.log({f'seed_{FLAGS.seed[seed_id]}/train_return': episode_returns[seed_id]}, step=i)
                episode_returns[seed_id] = 0.0



        if i >= FLAGS.start_training:
            # if batch is None:
                # batch = replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, updates_per_step)
            # batch = replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, updates_per_step)
            # update_info = agent.update(batch)
            # batch = replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, updates_per_step)

            for _ in range(updates_per_step):
                batch = replay_buffer.sample_parallel(FLAGS.batch_size)
                update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                log_multiple_seeds_to_wandb(step=i, infos=update_info)
                for key in update_info:
                    for seed_id, value in enumerate(update_info[key]):
                        step_stats[seed_id][key] = value



        if i % FLAGS.eval_interval == 0:
            if algo == 'cel':
                if agent.tandem:
                    assert agent.n == 2
                    agent.set_eval_policy(eval_policy='single', eval_agent_id=0)
                    eval_stats_active = evaluate_parallel_seed_cel(agent, eval_env, FLAGS.eval_episodes)
                    agent.set_eval_policy(eval_policy='single', eval_agent_id=1)
                    eval_stats_passive = evaluate_parallel_seed_cel(agent, eval_env, FLAGS.eval_episodes)
                    eval_stats = np.empty(num_seeds, dtype=object)
                    for seed_id in range(num_seeds):
                        eval_stats[seed_id] = dict()
                        eval_stats[seed_id]['return'] = eval_stats_active[seed_id]['return']
                        for key, value in eval_stats_active[seed_id].items():
                            eval_stats[seed_id][f'active/{key}'] = value
                        for key, value in eval_stats_passive[seed_id].items():
                            eval_stats[seed_id][f'passive/{key}'] = value
                else:
                    agent.set_eval_policy(eval_policy='avg')
                    eval_stats_avg = evaluate_parallel_seed_cel(agent, eval_env, FLAGS.eval_episodes)
                    if agent.n > 1:
                        agent.set_eval_policy(eval_policy='random')
                        eval_stats_random = evaluate_parallel_seed_cel(agent, eval_env, FLAGS.eval_episodes)
                    else:
                        # No need to rerun, same stuff
                        eval_stats_random = eval_stats_avg
                    eval_stats = np.empty(num_seeds, dtype=object)
                    for seed_id in range(num_seeds):
                        eval_stats[seed_id] = dict()
                        eval_stats[seed_id]['return'] = eval_stats_avg[seed_id]['return']
                        for key, value in eval_stats_avg[seed_id].items():
                            eval_stats[seed_id][f'avg/{key}'] = value
                        for key, value in eval_stats_random[seed_id].items():
                            eval_stats[seed_id][f'random/{key}'] = value
            else:
                eval_stats = evaluate_parallel_seed(agent, eval_env, FLAGS.eval_episodes)

            for seed_id in range(num_seeds):
                for key, value in eval_stats[seed_id].items():
                    wandb.log({f'seed_{FLAGS.seed[seed_id]}/{key}': value}, step=i)

                eval_returns[seed_id].append(
                    (i, eval_stats[seed_id]['return']))
                np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed[seed_id]}.txt'),
                           eval_returns[seed_id],
                           fmt=['%d', '%.1f'])
                step_stats[seed_id].update(eval_stats[seed_id])
                # Convert all to json.
                for key, value in step_stats[seed_id].copy().items():
                    if hasattr(value, 'item'):
                        value = value.item()
                    step_stats[seed_id][key] = value
                stats[seed_id].append(step_stats[seed_id])
                with open(os.path.join(FLAGS.save_dir, f'{FLAGS.seed[seed_id]}_stats.json'), 'w') as f:
                    json.dump(stats[seed_id], f, indent=2)

                if FLAGS.save_agent:
                    assert False, 'Not implemented yet'
                    # agent.actor.save(os.path.join(FLAGS.save_dir, f'actor_{seed_id}'))
                    # agent.critic.save(os.path.join(FLAGS.save_dir, f'critic_{seed_id}'))
                    # agent.target_critic.save(os.path.join(FLAGS.save_dir, f'target_critic_{seed_id}'))
                    # agent.temp.save(os.path.join(FLAGS.save_dir, f'temp_{seed_id}'))
                    # TODO: maybe save opt state too
                if FLAGS.save_buffer:
                    assert False, 'Not implemented yet'
                    # replay_buffer.save(os.path.join(FLAGS.save_dir, f'buffer_{seed_id}'))

        if FLAGS.save_checkpoint and i % FLAGS.checkpoint_interval == 0:
            if i > FLAGS.checkpoint_interval:
                to_remove = i - FLAGS.checkpoint_interval
            else:
                to_remove = None
            save_checkpoint(FLAGS.save_dir,FLAGS.seed, i, replay_buffer, agent, stats, eval_returns, to_remove)


if __name__ == '__main__':
    app.run(main)
