docker run --gpus device="$CUDA_VISIBLE_DEVICES" -it \
      --ipc=host \
      --volume=$(pwd):/app \
      --volume="$HOME"/work/MuCon/root/:/data/root/ \
      --volume="$HOME"/work/MuCon/datasets/:/data/datasets/ \
      mucon:pytorch1.1 \
      "${@:1}"
