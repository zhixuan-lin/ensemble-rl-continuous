import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'sac_wd'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_size = 256
    #config.hidden_dims = (config.hidden_size, config.hidden_size)

    config.discount = 0.99
    config.discount_min = 0.99
    config.discount_period = 0
    config.discount_schedule = 'none'
    config.up_ratio = 0.5

    config.l2 = 0.
    config.fn_critic = False
    config.fn_actor = False
    config.ln_critic = False
    config.ln_params_critic = False
    config.weight_decay = 0.0
    config.wd_bias = False
    config.wd_critic = False
    config.wd_actor = False
    config.wd_target = False
    config.wd_ln = True
    config.wd_last = True
    config.wd_last_only = False
    config.wd_first_only = False
    config.wd_second_only = False
    config.wd_action_only = False
    config.wd_obs_only = False
    config.wd_decoupled = True
    config.soft_reset = False
    config.tanh_log_std = True
    config.backup_entropy = True

    config.tau = 0.005
    config.target_update_period = 1
    config.updates_per_step = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.policy_update_delay = 1

    config.replay_buffer_size = None
    config.max_norm_clip = False
    config.max_norm = 3.0
    config.max_norm_final = 3.0
    config.mn_schedule = 'none'
    config.mn_last = True
    config.td3_cdq = False


    return config
