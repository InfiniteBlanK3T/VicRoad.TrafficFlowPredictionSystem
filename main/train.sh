python src/train_model.py --model gru --n_scats 5
python src/train_model.py --model lstm --n_scats 5
python src/train_model.py --model saes --n_scats 5
python src/train_model.py --model cnnlstm --n_scats 5
python src/train_model.py --model bilstm --n_scats 5

## Careful with the following commands, they will take a lot of time to run
# python src/train_model.py --model gru --n_scats all
# python src/train_model.py --model lstm --n_scats all
# python src/train_model.py --model saes --n_scats all
# python src/train_model.py --model cnnlstm --n_scats all
# python src/train_model.py --model bilstm --n_scats all

spd-say "training done"
notify-send "training done"