### Recurrent MobileNet

#### Check Params
We propose Recurrent MobileNet for lightweight denoising, which the number of 
total parameters is under 100k. To check the number of model parameters, you could run

```python recurrent_mobilenet.py```

due to the megengine repeatedly counts the number of parameters in a recurrent network, 
we set the repeat times to one ( i.e.`unroll=1`) here.

#### Testing
```python test.py --path path_to_dataset```

it should generate a `result.bin` to current directory. The `path_to_dataset` is organized
the same as downloaded:
```
path_to_dataset
|--burst_raw
   |--competition_train_input.0.2.bin
   |--competition_train_gt.0.2.bin
   |--competition_test_input.0.2.bin
```

#### (Validating)
We also divide the last 1024 pairs in the training set as our validation set, you can 
also validate on our dataset.
```
python validate.py --path path_to_dataset
```