# A-Joint-Learning-Model-for-Click-Through-Rate-Prediction-in-Display-Advertising
## AT-RESNET：
This is the code of the paper "A Joint Learning Model for Click-Through Rate Prediction in Display Advertising".

## Requirements:
python3.7，tensorflow 1.10


## Instructions：
The IPinYou data set used in this paper comes from make-ipinyou-data https://github.com/Atomu2014/make-ipinyou-data Copy the train.txt and test.txt of the corresponding advertiser to the output/pnn directory.


The instruction of commands has been clearly stated in the codes (see the parse_args function).
The current implementation supports two tasks: regression and binary classification. The regression task optimizes RMSE, and the binary classification task optimizes Log Loss.
