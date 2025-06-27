# PileGNN
Automated design of bridge single-pile based on graph neural networks

![graph abstract.png](other/abstract.png)

# Lib Name-Version
python                    3.6.13  
pytorch                   1.7.1   
numpy                     1.19.2   
PyTorch Geometric         1.7.2

# dataset
1. The directory datasets/Pile_EB stores the partial datasets for end-bearing pile samples.
2. The directory datasets/Pile_F stores the partial datasets for friction pile samples.

Due to intellectual property restrictions, only a portion of the dataset is provided in the data directory for demonstration purposes.

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

# Result
Partial visualized test results are available in the result_display directory.
