python cli.py --method our_method \
  --train_iterations 10 \
  --data_dir ./ \
  --model_type xlm-roberta \
  --model_name_or_path xlm-roberta-large \
  --task_name parsinlu-movie-sentiment \
  --output_dir ./new_model_check \
  --pet_max_seq_length 256