# MT-DENet
## Code for the paper "MT-DENet: Prediction of Post-therapy OCT Images in Diabetic Macular Edema by Multi-Temporal Disease Evolution Network"


## Requirements
- Python 3.8 + is recommended.
- Pytorch 1.13.0 + is recommended.
- Dgl 1.1.2 is recommended.
- This code is tested with CUDA 11.7 toolkit and CuDNN 8.5.0.

## Data Preparation
### File Organization

``` 
├── [Your Path]
    ├── DME
        ├── Monthly
            ├── 0210MOD-000000.npy,0210MOD-000001.npy,xxx
        └── GILA
            ├── 0204GOS-000000.npy,0204GOS-000001.npy,xxx
        └── TREX
            ├── 0220TOD-000000.npy,0220TOD-000001.npy,xxx
```
Each *.npy file contains data from five follow-up visits (t1, t2, t3, t4, t5). Each follow-up consists of a combination of seven consecutive B-scan images.

## Training and Testing

### Training
```
python train.py \
--data_path [path to the dataset]
--train_list [train set list]
--test_list [test set list]
--result_dir [path to saving logs and models]
```

### testing
```
python test.py \
--data_path [path to the dataset]
--test_list [test set list]
--model_path [path to trained model]
--results_dir [path to saving images]
```

### output
- SSIM, LPIPS, PSNR
- Real OCT B-scan of t5
- Predicted OCT B-scan of t5
