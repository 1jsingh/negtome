accelerate launch distributed_inference_diversity_sdxl.py \
  --batch_size=8 \
  --use_cads=False \
  --use_negtome=True \
  --merging_alpha=.9 \
  --merging_threshold=0.5 \
  --merging_t_start=1000 \
  --merging_t_end=900 \
  --output_pkl='./output/negtome-sdxl-threshold0.5.pkl'


accelerate launch distributed_inference_diversity_sdxl.py \
  --batch_size=8 \
  --use_cads=False \
  --use_negtome=True \
  --merging_alpha=.9 \
  --merging_threshold=0.7 \
  --merging_t_start=1000 \
  --merging_t_end=900 \
  --output_pkl='./output/negtome-sdxl-threshold0.7.pkl'


  accelerate launch scripts.distributed_inference_diversity_sdxl \
    --batch_size=8 \
    --num_images_per_prompt=4 \
    --coco_captions=./data/coco2017_val/annotations/captions_val2017.json \
    --max_num_captions=16 \
    --use_cads=False \
    --use_negtome=False \
    --output_pkl='./output/base-sdxl-coco.pkl'

accelerate launch scripts/distributed_inference_diversity_sdxl.py \
  --batch_size=8 \
  --num_images_per_prompt=4 \
  --coco_captions=./data/coco2017_val/annotations/captions_val2017.json \
  --max_num_captions=1 \
  --max_num_prompts=16 \
  --use_cads=False \
  --use_negtome=False \
  --output_pkl='./output/base-sdxl-coco.pkl'

    export PYTHONPATH=$PYTHONPATH:'/data/home/jaskirats/project/alphagen/negtome'
