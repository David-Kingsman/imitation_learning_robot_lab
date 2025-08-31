python ./imitation_learning_robot_lab/scripts/rollout.py \
  --policy.path=outputs/models/act_pick_and_place/checkpoints/008000/pretrained_model \
  --env.type=pick_and_place \
  --policy.device=cuda \
  --policy.use_amp=false