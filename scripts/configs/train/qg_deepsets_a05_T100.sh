# Fine-tuned qg teacher (fine_tune_qg_pretrain_l) -> DeepSets/PFN student on
# qg. KD alpha=beta=0.5, T=100, size=small. Temperature sweep companion to
# qg_deepsets_a05.sh (T=4), qg_deepsets_a05_T1.sh (T=1), and
# qg_deepsets_a05_T16.sh (T=16).
#
# T=100 sits past the point where the teacher's soft targets stop changing:
# the logit gap has mean |dz| = 0.95 and p99 = 3.94, so dz/T < 0.04 for
# essentially every event and get_distill_loss is in its logit-matching limit.
# The run tests whether T still moves the student once the soft targets are
# effectively uniform.
#
# DeepSets is Phi-embed + masked pool + rho-MLP, no attention.
#
# Everything else -- batch 128, iterations 1000, epoch 50, lr 5e-4, wd 0.5,
# teacher companion dir/tag, data path -- comes from _defaults.sh unchanged.
OUTDIR=/pscratch/sd/m/mbenyas/OmniLearned/checkpoints/
ARCH=deep-sets
DATASET=qg
INTERACTION=1
LOCAL_INTERACTION=0
EXTRA_FLAGS="--use-pid"
DISTILL_T=100
TEACHER_TAG=fine_tune_qg_pretrain_l
SAVE_TAG=distill_qg_deepsets_small_scratch_a05_T100
