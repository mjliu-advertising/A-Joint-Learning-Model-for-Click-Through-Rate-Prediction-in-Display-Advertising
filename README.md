# A-Joint-Learning-Model-for-Click-Through-Rate-Prediction-in-Display-Advertising
AT-RESNET：
这是论文“A Joint Learning Model for Click-Through Rate Prediction in Display Advertising”的代码。

Requirements:
python3.7，tensorflow 1.10


使用方法：
本论文使用的IPinYou数据集来自make-ipinyou-data https://github.com/Atomu2014/make-ipinyou-data 将对应广告主的train.txt、test.txt复制到output/pnn目录即可


The instruction of commands has been clearly stated in the codes (see the parse_args function).
The current implementation supports two tasks: regression and binary classification. The regression task optimizes RMSE, and the binary classification task optimizes Log Loss.
