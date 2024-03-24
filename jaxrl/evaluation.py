from typing import Dict, List

import flax.linen as nn
import gym
import numpy as np
from collections import defaultdict


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int, use_agent_state: bool = False) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            if use_agent_state:
                action = agent.sample_actions(
                    observation, temperature=0.0, eval_mode=True
                )
            else:
                action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats

def evaluate_and_record(agent: nn.Module, env: gym.Env,
             num_episodes: int, use_agent_state: bool = False) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    trajectories = []
    for _ in range(num_episodes):
        trajectories_temp = []
        observation, done = env.reset(), False
        while not done:
            if use_agent_state:
                action = agent.sample_actions(
                    observation, temperature=0.0, eval_mode=True
                )
            else:
                action = agent.sample_actions(observation, temperature=0.0)
            next_observation, reward, done, info = env.step(action)
            if not done or 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
                mask = 1.0
            else:
                mask = 0.0
            trajectories_temp.append((observation, action, reward, done, mask, next_observation))
            observation = next_observation
        # Append finished episode
        trajectories.append(trajectories_temp)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats, trajectories

def evaluate_parallel_seed(agent: nn.Module, env: gym.vector.VectorEnv,
             num_episodes: int) -> Dict[str, float]:
    assert isinstance(env, gym.vector.VectorEnv)
    stats = np.empty(env.num_envs, dtype=object)
    for env_id in range(env.num_envs):
        stats[env_id] = {'return': [], 'length': []}
    total_episodes = np.zeros(env.num_envs)
    observation, done = env.reset(), False
    while True:
        action = agent.sample_actions(observation, temperature=0.0)
        observation, _, done, info = env.step(action)

        for i in range(env.num_envs):
            if done[i] and total_episodes[i] < num_episodes:
                for k in stats[i].keys():
                    stats[i][k].append(info['episode'][i][k])

                total_episodes[i] += 1
        if (total_episodes >= num_episodes).all():
            break


    for i in range(env.num_envs):
        for k, v in stats[i].items():
            assert len(stats[i][k]) == num_episodes
            stats[i][k] = np.mean(v)

    return stats

def evaluate_parallel_seed_cel(agent: nn.Module, env: gym.vector.VectorEnv,
             num_episodes: int) -> Dict[str, float]:
    assert isinstance(env, gym.vector.VectorEnv)
    stats = np.empty(env.num_envs, dtype=object)
    for env_id in range(env.num_envs):
        stats[env_id] = defaultdict(list)
    total_episodes = np.zeros(env.num_envs)
    observation, done = env.reset(), False
    agent.begin_episode(mask=np.ones(env.num_envs, dtype=bool), eval_mode=True)

    agent_info_list = np.empty(env.num_envs, dtype=object)
    for env_id in range(env.num_envs):
        agent_info_list[env_id] = defaultdict(list)
    while True:
        action, agent_info = agent.sample_actions(observation, eval_mode=True)
        for key in agent_info:
            assert agent_info[key].shape == (env.num_envs,)
            for env_id in range(env.num_envs):
                agent_info_list[env_id][key].append(agent_info[key][env_id])

        observation, _, done, info = env.step(action)

        for i in range(env.num_envs):
            if done[i] and total_episodes[i] < num_episodes:
                # for k in ['return', 'length']:
                for k in ['return']:
                    stats[i][k].append(info['episode'][i][k])
                    # Per agent log
                    if agent.eval_policy == 'random':
                        agent_id = int(agent.eval_agent_id[i])
                        stats[i][f'{k}_{agent_id}'].append(info['episode'][i][k])

                total_episodes[i] += 1
        if (total_episodes >= num_episodes).all():
            break

        if done.any():
            # assert done.all()
            # Resample when necessary
            agent.begin_episode(mask=done, eval_mode=True)


    for i in range(env.num_envs):
        if agent.eval_policy == 'random':
            # Sanity check
            total = sum([len(stats[i][f'return_{agent_id}']) for agent_id in range(agent.n) if f'return_{agent_id}' in stats[i]])
            assert total == num_episodes, (total, num_episodes)
        for k, v in stats[i].items():
            # if k in ['return', 'length']:
            if k in ['return']:
                assert len(stats[i][k]) == num_episodes
                
            stats[i][k] = np.mean(v)
        for k, v in agent_info_list[i].items():
            # Hard coded
            # assert len(v) == num_episodes * 1000
            stats[i][k] = np.mean(v)
    return stats

def evaluate_parallel(agent: nn.Module, env: gym.vector.VectorEnv,
             num_episodes: int, use_agent_state: bool = False) -> Dict[str, float]:
    assert isinstance(env, gym.vector.VectorEnv)
    stats = {'return': [], 'length': []}
    successes = None

    total_episodes = 0
    observation, done = env.reset(), False
    complete = False
    while True:
        if use_agent_state:
            action = agent.sample_actions(
                observation, temperature=0.0, eval_mode=True
            )
        else:
            action = agent.sample_actions(observation, temperature=0.0)
        observation, _, done, info = env.step(action)

        for i in range(env.num_envs):
            if done[i]:
                for k in stats.keys():
                    stats[k].append(info['episode'][i][k])

                if 'is_success' in info:
                    if successes is None:
                        successes = 0.0
                    successes += info['is_success'][i]
                total_episodes += 1
            if total_episodes >= num_episodes:
                complete = True
                break
        if complete:
            break


    for k, v in stats.items():
        assert len(stats[k]) == num_episodes
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats

def evaluate_parallel_and_record(agent: nn.Module, env: gym.vector.VectorEnv,
             num_episodes: int, use_agent_state: bool = False) -> Dict[str, float]:
    assert isinstance(env, gym.vector.VectorEnv)
    stats = {'return': [], 'length': []}
    successes = None

    total_episodes = 0
    observation, done = env.reset(), False
    complete = False
    trajectories = []
    trajectories_temp = [[] for i in range(env.num_envs)]
    while True:
        if use_agent_state:
            action = agent.sample_actions(
                observation, temperature=0.0, eval_mode=True
            )
        else:
            action = agent.sample_actions(observation, temperature=0.0)
        next_observation, reward, done, info = env.step(action)

        for i in range(env.num_envs):
            if done[i]:
                # This thing is needed to compute entropy reward
                real_next_observation = info['final_observation'][i]
                if 'TimeLimit.truncated' in info and info['TimeLimit.truncated'][i]:
                    mask = 1.0
                else:
                    mask = 0.0
            else:
                mask = 1.0
                real_next_observation = next_observation[i]
            trajectories_temp[i].append((observation[i], action[i], reward[i], done[i], mask, real_next_observation))
            if done[i]:
                # Append finished episode
                trajectories.append(trajectories_temp[i])
                # Clear episode
                trajectories_temp[i] = []
                for k in stats.keys():
                    stats[k].append(info['episode'][i][k])

                if 'is_success' in info:
                    if successes is None:
                        successes = 0.0
                    successes += info['is_success'][i]
                total_episodes += 1
            if total_episodes >= num_episodes:
                complete = True
                break
        observation = next_observation
        if complete:
            break


    for k, v in stats.items():
        assert len(stats[k]) == num_episodes
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats, trajectories

def caculate_estimation_bias(agent: nn.Module, trajectories: List, first_n_if_truncated: int = 100):
    """As the function name says.

    WARNING: if the episode is truncated we only take the first few states for
    computing that. Must be careful with what you used 

    for DMC with action repeat=2, 100 is a sound value. for AR=4 it is ok.
    """
    all_mc_returns = np.zeros(0)
    all_entropy_returns = np.zeros(0)
    all_values = np.zeros(0)
    for trajectory in trajectories:
        observations = []
        next_observations = []
        actions = []
        rewards = []
        masks = []
        dones = []
        for (obs, action, reward, mask, done, next_observation) in trajectory:
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            masks.append(mask)
            dones.append(done)
            next_observations.append(next_observation)
        observations = np.stack(observations, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0)
        masks = np.stack(masks, axis=0)
        dones = np.stack(dones, axis=0)
        next_observations = np.stack(next_observations, axis=0)

        L = observations.shape[0]
        truncated = bool(mask)

        # compute value estimates and entropy
        # (L,), (L,)
        q_values, entropy_rewards = agent.get_q_value_and_entropy_reward(observations, actions, next_observations)

        # Compute monte carlo rollout and entropy return
        mc_returns = np.zeros_like(q_values)
        entropy_returns = np.zeros_like(q_values)
        last_mc_return = 0.0
        last_entropy_return = 0.0
        for t in reversed(range(L)):
            last_mc_return = rewards[t] + agent.discount * last_mc_return
            last_entropy_return = agent.discount * (entropy_rewards[t] + last_entropy_return)
            mc_returns[t] = last_mc_return
            entropy_returns[t] = last_entropy_return

        if truncated:
            # Only use the first part the trajectory
            mc_returns = mc_returns[:first_n_if_truncated]
            entropy_returns = entropy_returns[:first_n_if_truncated]
            q_values = q_values[:first_n_if_truncated]

        all_mc_returns = np.append(all_mc_returns, mc_returns)
        all_entropy_returns = np.append(all_entropy_returns, entropy_returns)
        all_values = np.append(all_values, q_values)



    mc_return_mean = all_mc_returns.mean().item()
    entropy_return_mean = all_entropy_returns.mean().item()
    value_mean = all_values.mean().item()
    q_bias = all_values - (all_mc_returns + all_entropy_returns)
    q_bias_rmse = np.sqrt((q_bias ** 2).mean()).item()
    q_bias_abs = np.abs(q_bias).mean().item()
    # q_bias_normalized = (q_bias / np.abs(mc_return_mean + entropy_return_mean))
    q_bias_mean = q_bias.mean().item()
    # q_bias_std = q_bias.std().item()
    # q_bias_normalized_mean = q_bias_normalized.mean().item()
    # q_bias_normalized_std = q_bias_normalized.std().item()

    return mc_return_mean, entropy_return_mean, value_mean, q_bias_mean, q_bias_rmse, q_bias_abs

