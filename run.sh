 /Users/boss/anaconda3/bin/python3 ./run_classifier.py \
  --task_name=text_clf \
  --do_train=true \
  --do_eval=true \
  --do_train_and_eval=false \
  --do_predict=false \
  --data_dir=./tnews_public/ \
  --save_checkpoints_steps=50 \
  --vocab_file=./vocab.txt \
  --bert_config_file=./bert_config.json \
  --init_checkpoint=./model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --num_train_epochs=5 \
  --output_dir=./output/
