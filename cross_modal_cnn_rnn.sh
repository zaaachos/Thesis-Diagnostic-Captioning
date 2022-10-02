python3 dc.py \
---dataset iu_xray \
--model_choice cnn_rnn \
--max_length 40 \
--image_encoder efficientnet0 \
--sample_method beam_3 \
--batch_size 8 \
--n_gpu 1 \
--epochs 50 \
--dropout 0.2