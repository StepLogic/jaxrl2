import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.cnn_features = (8, 16, 32, 32)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 50
    config.encoder = "d4pg"
    config.discount = 0.98
    config.tau = 0.005
    config.init_temperature = 1.0
    config.target_entropy = None
    config.backup_entropy = True
    config.critic_reduction = "mean"

    return config
