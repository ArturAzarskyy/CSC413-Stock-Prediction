# CSC413 Stock Prediction project

Our project consists of mainly two parts the preprocessing of the data located in `transformer_prepros.ipynb`,
and the ML models would be located in `transformer_model.ipynb`.

## Pre-processing of the data

There were two possible datasets which we could have used `price-volume-data-for-all-us-stocks-etfs`(MARJ) 
or  `amex-nyse-nasdaq-stock-histories`(YEN) but we choose to go with YEN dataset since it provided file
which specified what stocks are provided, and did not put repeated stocks into multiple folders, and had
some files missing. 

When we get the data from Kaggle we create 3 files of which will contain data for validation, we  could have created one
file containing all the information but decided to split it into separate files to make it more clear where exactly the
information is saved to where. Each of the files contains two earrays  `data` and `labels` arrays. The earrays structure
was chosen due to the fact that this dataset contains a lot of information, with current hyperparameters for
preprocessing of information it ends up being 12 million entries for training., and earrays allows us to grab a small
portion when training without loading full pre-processed dataset, and they make saving to the file and easy task as well.



Then we go through each line of `all_symbols.txt` which contains names of possible stocks provided in the dataset, though
there are few stocks which are in the text file but not present as a csv file. Then we load in the csv file using the
pandas and sort the data by date. After we sort the data we get perform moving avg if the hyperparameter is corresponding
hyperparameter is set to true. To make sure that we do not introduce too much noisy data from the stocks which existed only
for few days and weeks we have a check which makes sure that there is at least `stock_histoy_lenght + 1` number of days
available when generating the test set (which is 10% of the stock history).


After it was confirmed that the stock has enough data all the data is normalized by training max and min value, and 
saved then the data is saved to appropriate files and saving intervals of days with the label being the next day's
closing price. Then the files are saved locally first and then uploaded the zipped version of everything into the Google
Drive.




## Executing the Model

### Prep:


First we get the pre-processed zipped files from the Google Drive. Since we want to use PyTorch we need to define a custom 
Dataset class which given the name of the `.hdf5` file will make access to both `data` and `lables` directories of the file
available. We also introduce transformation function which transforms the numpy values into tensors, and we also have a
transformation which will help us to ensure that labels are of right shape if we decide to use RandomBatchSampler. I also
added a small benchmarking code which shows that the data is correctly loaded and benchmarking the spead of loading of 
both loaders

### Model:

##### Time2Vec:

We start off by implementing a method utilized in the paper [**Time2Vec: Learning a Vector Representation of Time**](https://arxiv.org/pdf/1907.05321.pdf). The main purposes of using Time2Vec is to capture the ideas of both periodic and non-periodic patterns and not running into the issue of time rescaling - measuring time using different scales (ex. days, hours seconds). For the first iteration, Time2Vec utilizes a linear or non-periodic representation of the time vector and every other iteration is a function of the time vector representing the periodic feature. In the paper, it was shown that the sin function outperformed other non-linear functions such as sigmoid, tanh and ReLU in terms of accuracy and stability.

![Time2Vec](https://user-images.githubusercontent.com/42477683/163737739-715d843d-fe11-4e79-9d45-a0f265324035.png)

Fot our initial model we decided to go with the simple transformer model. Which contains a multiple attention heads, encoder and decoder. As mentioned in the colab this model was enspired from original paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and Jan Schmitz's article 
[Stock predictions with state-of-the-art Transformer and Time Embeddings](https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6#:~:text=A%20Transformer%20is%20a%20neural,and%20Multi%2DHead%20Attention%20layer), 
to create our base model. Below you can see diagrams of One-head Attention archetecture, and Multi-head Attention archetecture we used for base model.


##### One-head Attention:
![OneHead](https://user-images.githubusercontent.com/42477683/163741343-6a4c54c7-68d0-4bdc-94fe-c71bb0f41ae5.png)

##### Multi-head Attention:
![MultiHead](https://user-images.githubusercontent.com/42477683/163741346-54c95a8f-20ff-40dc-9cf7-dfa51167af61.png)




### Results


### Ethical Consideration: 

It is important to understand that in order for one person to make money in the stock market and often times, another person typically loses money. Our model’s main purpose is to predict the future price of a stock. A successful stock market predictor can raise ethical issues if abused. This includes negatively impacting an excessive amount of buyers who take your loss or investing in companies that perform unethical business practices in orderto profit.  However, if used correctly, our model will help users develop a better overall understanding of factors affecting the stock market.

We are not obtaining our data from any insider unethical sources and are only using historical stock data to train our model. In that sense our data is 100% safe and legal to use. Users must consider that because our model is not 100% accurate, the model cannot guarantee any profits. Users are at risk if using this model to perform real world trades. 

## Contributions
- Artur Azarskyy: researched articles and papers on topic, created the pre-processing workflow, created the StockDataset
 class, and sampler which could be used to train the model, writing of the documentation of the parts I did. I refactored the
 time embeding to be a torch model, I also created all of the other parts related to the basic model.
- Jordan Tam: Researched articles and papers on topic, worked on Time2Vector, wrote about Ethical Consideration and the 
 other parts of that I worked on
- 
[//]: # (TODO: everyone please add your names and contributions here )
