# hvq_ql
conda env export --no-builds > full_env.yaml
conda env config vars set MUJOCO_GL=egl

mamba activate hvq
cd /home/yw4142/wm/hvq_ql
python pretrain_vqvae.py