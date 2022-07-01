CUDA_VISIBLE_DEVICES=5 python main.py --dataset="pomt" \
                                      --model="bert" \
                                      --batch_size=8 \
                                      --lr=3e-6\
                                      --label_num=2\
                                      --num_epochs=100\
                                      --snippet_length=200\
                                      --num_fold=5