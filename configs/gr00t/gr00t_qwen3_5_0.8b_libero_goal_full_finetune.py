# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GR00T VLA: Qwen3.5-0.8B (native hidden_size=1024) + linear projection
# 1024->2048 to match GR00T-N1.5 action head.

_qwen3_5_vla_ckpt = ('/limx/tos/limx_mani_checkpoints/checkpoints/mayer_GR00T/'
                     'gr00t_qwen3.5_0.8b_libero_all_0701_epoch24/checkpoints/'
                     'step-104160-epoch-24-loss=0.0312.safetensors')
_qwen3_5_tokenizer = (
    '/limx/tos/limx_mani_checkpoints/checkpoints/mayer_GR00T/'
    'gr00t_qwen3.5_0.8b_libero_all_0701_epoch24/tokenizer/')
_qwen3_5_vlm_config = {
    'architectures': ['Qwen3_5ForConditionalGeneration'],
    'attn_implementation': 'sdpa',
    'dtype': 'bfloat16',
    'image_token_id': 248056,
    'model_type': 'qwen3_5',
    'text_config': {
        'attention_bias':
        False,
        'attention_dropout':
        0.0,
        'attn_implementation':
        'sdpa',
        'attn_output_gate':
        True,
        'bos_token_id':
        None,
        'dtype':
        'bfloat16',
        'eos_token_id':
        248044,
        'full_attention_interval':
        4,
        'head_dim':
        256,
        'hidden_act':
        'silu',
        'hidden_size':
        1024,
        'initializer_range':
        0.02,
        'intermediate_size':
        3584,
        'layer_types': [
            'linear_attention', 'linear_attention', 'linear_attention',
            'full_attention', 'linear_attention', 'linear_attention',
            'linear_attention', 'full_attention', 'linear_attention',
            'linear_attention', 'linear_attention', 'full_attention',
            'linear_attention', 'linear_attention', 'linear_attention',
            'full_attention', 'linear_attention', 'linear_attention',
            'linear_attention', 'full_attention', 'linear_attention',
            'linear_attention', 'linear_attention', 'full_attention'
        ],
        'linear_conv_kernel_dim':
        4,
        'linear_key_head_dim':
        128,
        'linear_num_key_heads':
        16,
        'linear_num_value_heads':
        16,
        'linear_value_head_dim':
        128,
        'mamba_ssm_dtype':
        'float32',
        'max_position_embeddings':
        262144,
        'mlp_only_layers': [],
        'model_type':
        'qwen3_5_text',
        'mtp_num_hidden_layers':
        1,
        'mtp_use_dedicated_embeddings':
        False,
        'num_attention_heads':
        8,
        'num_hidden_layers':
        24,
        'num_key_value_heads':
        2,
        'pad_token_id':
        None,
        'partial_rotary_factor':
        0.25,
        'rms_norm_eps':
        1e-06,
        'rope_parameters': {
            'mrope_interleaved': True,
            'mrope_section': [11, 11, 10],
            'partial_rotary_factor': 0.25,
            'rope_theta': 10000000,
            'rope_type': 'default'
        },
        'tie_word_embeddings':
        True,
        'use_cache':
        True,
        'vocab_size':
        248320
    },
    'tie_word_embeddings': True,
    'transformers_version': '5.3.0',
    'video_token_id': 248057,
    'vision_config': {
        'attn_implementation': 'sdpa',
        'deepstack_visual_indexes': [],
        'depth': 12,
        'dtype': 'bfloat16',
        'hidden_act': 'gelu_pytorch_tanh',
        'hidden_size': 768,
        'in_channels': 3,
        'initializer_range': 0.02,
        'intermediate_size': 3072,
        'model_type': 'qwen3_5',
        'num_heads': 12,
        'num_position_embeddings': 2304,
        'out_hidden_size': 1024,
        'patch_size': 16,
        'spatial_merge_size': 2,
        'temporal_patch_size': 2
    },
    'vision_end_token_id': 248054,
    'vision_start_token_id': 248053
}
model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=_qwen3_5_vla_ckpt,
    name_mapping=None,
    strict_mapping=False,
    vlm_backbone=dict(
        type='Qwen3_5',
        vlm_backbone_id='qwen3_5_0.8b_pt',
        vlm_path=None,
        vlm_config=_qwen3_5_vlm_config,
        use_projection=True,
        projection_output_dim=2048,
        projection_type='linear',
        attn_implementation='sdpa'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        backbone_embedding_dim=2048,
        vl_self_attention_cfg=dict(
            attention_head_dim=64,
            num_attention_heads=32,
            num_layers=4,
            dropout=0.2,
            final_dropout=True,
            positional_embeddings=None),
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            num_attention_heads=32,
            cross_attention_dim=2048,
            num_layers=16,
            output_dim=1024,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            positional_embeddings=None),
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=32,
        ori_action_dim=7),
    freeze_vlm_backbone=False,
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=_qwen3_5_vla_ckpt,
    name_mapping=None,
    vlm_backbone=dict(
        type='Qwen3_5',
        vlm_backbone_id='qwen3_5_0.8b_pt',
        vlm_path=None,
        vlm_config=_qwen3_5_vlm_config,
        use_projection=True,
        projection_output_dim=2048,
        projection_type='linear',
        attn_implementation='sdpa'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        backbone_embedding_dim=2048,
        vl_self_attention_cfg=dict(
            attention_head_dim=64,
            num_attention_heads=32,
            num_layers=4,
            dropout=0.2,
            final_dropout=True,
            positional_embeddings=None),
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            num_attention_heads=32,
            cross_attention_dim=2048,
            num_layers=16,
            output_dim=1024,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            positional_embeddings=None),
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=32,
        ori_action_dim=7),
    freeze_vlm_backbone=False,
    freeze_projector=False)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='libero_goal_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[  # noqa: E251
                '/limx/tos/limx_mani_data/raw_data/LIBERO_lerobot/libero_goal_no_noops_lerobotv2.1',  # noqa: E501
            ],
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=2,
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.image',
                        'observation.images.wrist_image',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions']
                    }),
                dict(type='ParquetPrompter'),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_qwen3_5_tokenizer,
                    )),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='QWen2VLImageTransform',
                    min_pixels=56 * 56,
                    max_pixels=28 * 28 * 1280,
                    patch_size=16,
                    temporal_patch_size=2,
                    merge_size=2,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711]),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std')
            ],
            action_window_size=10,
            action_key='action',
            use_delta=False,
            statistic_name='libero_goal_no_noops',
            window_start_idx=0)))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=24,
    learning_rate=1.5e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_qwen3_5_tokenizer,
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks',
            'embodiment_ids', 'image_grid_thw'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.03,
    ),
    sharding_strategy='full-shard',
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_goal',
    model_family='pi0',
    eval_chunk_size=10,
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                embodiment_id=2,
                img_keys=['agentview_image', 'robot0_eye_in_hand_image']),
            dict(type='ConvertPILImageToNumpyArray'),
            dict(
                type='QWen2VLImageTransform',
                min_pixels=56 * 56,
                max_pixels=28 * 28 * 1280,
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.48145466, 0.4578275,
                            0.40821073],  # OPENAI_CLIP_MEAN
                image_std=[0.26862954, 0.26130258,
                           0.27577711],  # OPENAI_CLIP_STD
                img_key='pixel_values',
                to_tensor=True),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_qwen3_5_tokenizer,
                )),
            dict(
                type='LiberoProprioFromInputs',
                state_dim=64,
                norm_type='mean_std',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                out_key='states'),
        ]),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='mean_std',
    ),
)
