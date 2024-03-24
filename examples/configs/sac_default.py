import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'sac'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_size = 256
    #config.hidden_dims = (config.hidden_size, config.hidden_size)

    config.discount = 0.99

    config.l2 = 0.

    config.tau = 0.005
    config.target_update_period = 1
    config.updates_per_step = 1

    config.init_temperature = 1.0
    config.target_entropy = None

    config.replay_buffer_size = None
    config.tanh_log_std = True
    config.policy_update_delay = 1

    return config
