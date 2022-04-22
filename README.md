# SADeepcry
An end-to-end deep learning model based on the optimized self-attention and auto-encoder modules for predicting protein crystallization propensity.


# Requirements
* python == 3.6
* pytorch == 1.6
* Numpy == 1.16.2
* scikit-learn == 0.21.3


# Files:

1.data (Save at https://zenodo.org/record/6475529)

MF_DS: A specially curated benchmark dataset for the production of protein material stage. 

PF_DS: A specially curated benchmark dataset for the protein purification stage.

CF_DS: A specially curated benchmark dataset for the production of protein crystals stage.

CRYS_DS: A specially curated benchmark dataset for the protein crystallization stage.

BD_MCRYS: A specifically curated benchmark dataset for membrane protein crystallization propensity prediction.

The above five benchmark datasets are all collected from the TargetTrack database. Each benchmark dataset includes 8 files:

TE_feature.csv: 9319-dimensional artificial features for samples in the testing set.

TR_feature.csv: 9319-dimensional artificial features of samples in the training set.

TE_Sequence.fasta: Names and amino acid sequences of samples in the testing set.

TR_Sequence.fasta: Names and amino acid sequences of samples in the training set.

TE_Label: Labels for samples in the testing set.

TR_Label: Labels for samples in the training set.

model.pkl and vae.pkl: The modules of the pretrained SADeepcry model.

2.Code

Network: The network structure of SADeepcry.

main.py: This function can predict the protein crystallization propensity of the given proteins using a pretrained model.


# Train and test folds
python main.py --rawdata_dir /Your path 

rawdata_dir: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

All files of Data and Code should be stored in the correct folder to run the model.

Example:

```bash
python main.py --rawdata_dir /data
```
# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Haochen Zhao at zhaohaochen@csu.edu.cn.
