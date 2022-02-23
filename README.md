# FashionSenti
This is the official data & code for the WSDM21 paper **Sentiment Analysis of Fashion Related Posts in Social Media**. [paper](https://arxiv.org/abs/2111.07815)

## Data
Our dataset can be downloaded via the [link](https://drive.google.com/file/d/1Uk1p7MjxqgzGHE9Pja4iIC8VMH-OlUfs/view?usp=sharing)

In the folder, final_dict_{train/test/val}.pkl is the caption corresponding to each post.

attribute_dict1.pkl is the attribute dict of each fashion item in the post.

img_feature/face_dict.pkl is the face area information of the person in each post.

img_feature/item_dict.pkl is the fashion item area information in each post.

## Code
Model/ stores the code of the baselines and the different components

Preprocess/ stores the basic preprocess code

opt/ stores the evaluation code

## Cite
```
@inproceedings{Yuan2021SentimentAO,
  title={Sentiment Analysis of Fashion Related Posts in Social Media},
  author={Yifei Yuan and Wai Lam},
  booktitle={WSDM},
  year={2021}
}
```
