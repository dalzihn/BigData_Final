<h2 align="center"><strong>University of Economics Ho Chi Minh 

Big Data and Applications 
</strong></h2>

<p align="center">
  <img src="./img/UEH-Tech School.png" width="500">
</p>


# Group 09, Exploring Apache Spark's performance and its application in mining research topics on COVID-19

This repo represents the final project of group 09 for the course *Big Data and Applications*. In this project, we provide an in-depth analysis of Spark's performance based on time and throughput in comparison with NTLK and Scikit-learn. Additionally, we will also train and apply LDA model to discover latent topics related to COVID-19 using [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) on Kaggle.


## Repo overview
The folder structure of this repo is as follows:
```
└── 📁code
    └── evaluation.ipynb
    └── helper.py
    └── metrics.py
└── 📁data
    └── 📁merged
    └── 📁processed
└── 📁img
└── 📁model
```

The repo is divided into four folders, the `code/` folder contains the python code for the project. In detail, `evaluation.ipynb` is the notebook used to perform evaluation and discover latent topics, while `helper.py` contains essential functions for preprocessing and training preprocessing models, and `metrics.py` stores code for calculating metrics such as PUW.

The `data/` folder consists of two subfolders, `merged` and `processed`, `merged` contains 10000 papers extracted from [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) divided into three .json files, while `proceessed` is the TF-IDF feature of 10000 papers

## Getting started


## Results

## Deployment

## Contributing

## Acknowledgements