CUDA_VISIBLE_DEVICES=1 python main.py --dataset="snes" \
                                      --model="mac" \
                                      --batch_size=32 \
                                      --lr=0.0001\
                                      --label_num=2\
                                      --lstm_layers=1\
                                      --num_att_heads_for_words=5\
                                      --num_att_heads_for_evds=2\
                                      --snippet_length=100\
                                      --hidden_size=300\
                                      --num_epochs=100\
                                      --num_fold=5\
                                      --embedding='bert'\
                                      --seed=12345\
                                      --up_bound=0
                                   