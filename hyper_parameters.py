import torch.nn as nn

# "env_name": {
#         "model_id": {
#             "seed": 0,
#             "epochs": 50,
#             "steps_per_epoch": 4000,
#             "max_episode_length": 1000,
#             "gradient_actor_loss_max_steps": 80,
#             "gradient_critic_loss_max_steps": 80,
#             "agent_hidden_sizes": (64, 64),
#             "agent_activation_function": nn.Tanh,
#             "agent_actor_learning_rate": 3e-4,
#             "agent_critic_learning_rate": 1e-3,
#             "agent_std": 0.5,
#             "gamma": 0.99,
#             "gae_lambda": 0.97,
#             "clip_ratio": 0.2,
#             "agent_return_update_interval": 100,
#             "solved_environment_average_return": 195,
#             "break_training_if_solved": False
#         }
#     }

config = {
    "general_configs": {
        "models_dir": "./models",
        "tensorboard_dir": "./runs",
        "debug_dir": "./debugs"
    },
    "defaults": {
        "seed": 0,
        "epochs": 50,
        "steps_per_epoch": 2000,  # 20 * max_episode_steps => 20 episodes in total
        "max_episode_steps": 100,
        "gradient_actor_loss_max_steps": 80,
        "gradient_critic_loss_max_steps": 80,
        "agent_state_layer": (nn.Linear, nn.Tanh, {}),  # (layer, activation fct of the state NN layer)
        "agent_action_layer": (),  # (layer, activation fct of the action NN layer)
        # types of state and actions NNs (Linear by default)
        "agent_hidden_layers": [(nn.Linear, 64, nn.Tanh, {}), (nn.Linear, 64, nn.Identity, {})],
        # types and sizes of hidden NNs
        "agent_actor_learning_rate": 3e-4,
        "agent_critic_learning_rate": 1e-3,
        "agent_std": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "clip_ratio": 0.2,
        "break_training_if_solved": True,
        "reward_threshold": None
    },
    'CartPole-v0': {
        'defaults': {
            'epochs': 20,
            "break_training_if_solved": False,
        }
    },
}
