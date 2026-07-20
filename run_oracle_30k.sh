CUDA_VISIBLE_DEVICES=1 python main.py +name=oracle_30k algorithm=latent_world_model \
  algorithm.training_stage=2 algorithm.action_dim=8 algorithm.latent_dim=512 \
  algorithm.load_ae=outputs/2026-06-28/11-10-12/checkpoints/stage1_best.ckpt \
  algorithm.sampling_strategy=terminal_only algorithm.noise_scheduler.loss_weighting=uniform \
  dataset=real_aloha_dataset dataset.action_mode=bimanual_rope \
  dataset.dataset_dir=/home/rl5372/Video-driven-embedded-system/data/partial_100ep_resplit/bimanual_rope \
  dataset.horizon=10 dataset.obs_keys=[camera_0_color] dataset.val_horizon=20 \
  experiment=exp_latent_dyn experiment.training.batch_size=4 \
  experiment.training.max_steps=30000 experiment.validation.val_every_n_step=3000 \
  experiment.training.checkpointing.every_n_train_steps=5000 \
  experiment.training.log_every_n_steps=50 \
  wandb.entity=rl5372-new-york-university \
  algorithm.use_diff_cond=True
