# PileGNN
Automated design of bridge single-pile based on graph neural networks

![graph abstract.png](other/abstract.png)

# Lib Name-Version
python                    3.6.13  
pytorch                   1.7.1   
numpy                     1.19.2   
PyTorch Geometric         1.7.2

# dataset
1. The directory `data/data_EB` stores the datasets for end-bearing pile samples.  
2. The directory `data/data_F` stores the datasets for friction pile samples.  
3. For training sets:  
   - `data/data_EB/train_A` and `data/data_F/train_A` store the **source features**.  
   - `data/data_EB/train_B` and `data/data_F/train_B` store the **labels**.  
4. For testing sets:  
   - `data/data_EB/test_A` and `data/data_F/test_A` store the **source features**.  
   - `data/data_EB/test_B` and `data/data_F/test_B` store the **labels**.  
5. Model prediction results (outputs) are stored in:  
   - `result/Pile_EB` for end-bearing pile samples.  
   - `result/Pile_F` for friction pile samples.  

**Note: Due to project privacy and intellectual property considerations, only a representative subset of the training dataset is publicly shared for reference.**


# Datasets directory
data/  
├── data_EB/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# End-bearing pile datasets  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── cond/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# input parameters  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train/   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── test/   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_A/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# inputs  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_B/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# labels  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── test_A/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# inputs  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── test_B/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# labels  
└── data_F/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Friction pile datasets  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── cond/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# input parameters  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train/   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── test/   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_A/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# inputs  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_B/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# labels  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── test_A/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# inputs  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── test_B/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# labels  


# Training
To train models using end-bearing pile.
```bash
   python train_EB.py
```
To train models using friction pile.
```bash
   python train_F.py
```
# Inference
The trained model weight files (.pth) are stored in the checkpoint directory.

To test models using end-bearing pile
```bash
   python test_EB.py
```

To test models using friction pile
```bash
   python test_F.py
```


# Result_PNG
Partial visualized test results are available in the directory result/PNG.