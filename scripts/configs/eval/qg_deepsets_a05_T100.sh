# Distilled DeepSets-small qg student (distill_qg_deepsets_small_scratch_a05_T100), test split.
CHECKPOINT_DIR=/pscratch/sd/m/mbenyas/OmniLearned/checkpoints/
EVAL_ROOT=/pscratch/sd/m/mbenyas/OmniLearned/eval
ARCH=deep-sets
DATASET=qg
NUM_CLASSES=2
SIZE=small
INTERACTION=1
LOCAL_INTERACTION=0
EXTRA_FLAGS="--use-pid"
SAVE_TAG=distill_qg_deepsets_small_scratch_a05_T100
