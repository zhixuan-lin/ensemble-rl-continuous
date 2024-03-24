import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'cel'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.n = 10

    config.hidden_size = 256
    #config.hidden_dims = (config.hidden_size, config.hidden_size)

    config.discount = 0.99

    config.policy_update_delay = 1
    config.updates_per_step = 1
    # config.l2 = 0.

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.backup_entropy = True

    config.replay_buffer_size = 1000000
    config.cel = False
    config.cel_type = 'q'
    config.tandem = False
    config.active_prob = 0.5
    config.aux_huber = False
    config.huber_delta = 1.0

    config.ln_critic = False
    config.ln_params_critic = True

    return config
