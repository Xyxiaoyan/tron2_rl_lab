from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg

import os
robot_type = os.getenv("ROBOT_TYPE")

#-----------------------------------------------------------------
@configclass
class SF_TRON2AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "sf_tron_2a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )

#-----------------------------------------------------------------
@configclass
class WF_TRON2AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "wf_tron_2a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class SF_TRON2ACampPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_camp"


#-----------------------------------------------------------------
@configclass
class SF_TRON2ACampDistillFineTunePPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    """Low-drift PPO settings for adapting a distilled policy to Camp."""

    # This is deliberately a short, frequently checkpointed adaptation run.
    # With 4096 environments, 24 steps already provide 98,304 samples/update.
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "sf_tron_2a_camp_distill_finetune"

    # Used only without a checkpoint. A resumed checkpoint restores its own
    # learned action standard deviation.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # Limit policy drift: the online PPO signal learns command response and
    # terrain transitions while imitation replay preserves specialist gaits.
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.001,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        obs_history_len=10,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON2ACampPPORunnerCfg(WF_TRON2AFlatPPORunnerCfg):
    experiment_name = "wf_tron_2a_camp"


#-----------------------------------------------------------------
@configclass
class SF_TRON2AStairsPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_stairs"


#-----------------------------------------------------------------
@configclass
class SF_TRON2AGapPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_gap"


#-----------------------------------------------------------------
# Unified-interface teacher policies used by multi-teacher distillation.
@configclass
class SF_TRON2AContinuousTeacherPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_teacher_continuous"


@configclass
class SF_TRON2AStairsTeacherPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_teacher_stairs"


@configclass
class SF_TRON2AObstacleTeacherPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_teacher_obstacle"


@configclass
class SF_TRON2AGapTeacherPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_teacher_gap"


@configclass
class SF_TRON2AMultiTeacherRunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    experiment_name = "sf_tron_2a_multi_teacher"
    max_iterations = 5000


#-----------------------------------------------------------------
@configclass
class WF_TRON2AStairsPPORunnerCfg(WF_TRON2AFlatPPORunnerCfg):
    experiment_name = "wf_tron_2a_stairs"
