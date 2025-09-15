#! /bin/bash

MAX_EPOCHS=2000
mkdir -p logs/1a-rcp-time-series-batch

python 1a-rcp-time-series-batch.py --model_name RNN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/RNN0.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name GRU --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/GRU0.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name LSTM --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/LSTM0.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name DenseINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/DenseINN0.log 2>&1 & 
python 1a-rcp-time-series-batch.py --model_name SparseINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/SparseINN0.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name VariableINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/VariableINN0.log 2>&1 &    

python 1a-rcp-time-series-batch.py --model_name RNN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/RNN1.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name GRU --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/GRU1.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name LSTM --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/LSTM1.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name DenseINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/DenseINN1.log 2>&1 & 
python 1a-rcp-time-series-batch.py --model_name SparseINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/SparseINN1.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name VariableINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/VariableINN1.log 2>&1 &    

python 1a-rcp-time-series-batch.py --model_name RNN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/RNN2.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name GRU --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/GRU2.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name LSTM --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/LSTM2.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name DenseINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/DenseINN2.log 2>&1 & 
python 1a-rcp-time-series-batch.py --model_name SparseINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/SparseINN2.log 2>&1 &
python 1a-rcp-time-series-batch.py --model_name VariableINN --max_epochs ${MAX_EPOCHS} > logs/1a-rcp-time-series-batch/VariableINN2.log 2>&1 &    
