# CSC413 Project - Stock Prediction

This project explores the task of predicting the price of stocks via a Neural Network with the Transformer architecture.

The preprocessing of the data located in `transformer_prepros.ipynb`, and the ML models are located in `transformer_model.ipynb`.

## Pre-processing of the Data

There were two possible datasets which we considered: `price-volume-data-for-all-us-stocks-etfs`(MARJ) 
or  `amex-nyse-nasdaq-stock-histories`(YEN). We chose to use the YEN dataset since it provided a file
which specified what stocks are provided, and did not put repeated stocks into multiple folders, and had
some files missing. 

When we get the data from Kaggle, we create 3 files of which will contain data for training, validation and testing. We considered creating a single file containing all the information but we decided to split it into separate files to make it more clear where information was saved to. 

Each of the files contains two earrays `data` and `labels` arrays. The earrays structure
was chosen because this dataset contains a large quantity of information-- with current hyperparameters for
preprocessing of information it ends up being 12 million entries for training. Earrays allows us to grab a small
portion when training without loading full pre-processed dataset, and they make saving to the file an easy task.

Next, we go through each line of `all_symbols.txt` which contains names of possible stocks provided in the dataset, though
there are few stocks which are in the text file but not present as a `.csv` file. Then we load in the `.csv` file using Pandas and sort the data by date. After we sort the data we perform a moving average if the hyperparameter is corresponding
hyperparameter is set to true. To make sure that we do not introduce too much noisy data from the stocks which existed only
for few days and weeks we have a check which makes sure that there is at least `stock_history_length + 1` number of days
available when generating the test set (which is 10% of the stock history).

After it was confirmed that the stock has enough data, all the data is normalized by training max and min value, and 
saved. The data is saved to appropriate files and saving intervals of days with the label being the next day's
closing price. Then the files are saved locally first and then uploaded the zipped version of everything into the Google
Drive.

## Executing the Model

### Preparation:

First we get the pre-processed zipped files from the Google Drive. Since we want to use PyTorch we need to define a 
StockDataset class which, given the name of the `.hdf5` file, will make access to both `data` and `lables` directories of the file
available. We also introduce transformation function which transforms the numpy values into tensors, and we also have a
transformation which will help us to ensure that labels are of right shape if we decide to use RandomBatchSampler. We also
included benchmarking code which shows that the data is correctly loaded can check the loading speed for both loaders. 

### Model:

##### Time2Vec:

We start off by implementing a method utilized in the paper [**Time2Vec: Learning a Vector Representation of Time**](https://arxiv.org/pdf/1907.05321.pdf). The main purpose of using Time2Vec is to capture the ideas of both periodic and non-periodic patterns and not running into the issue of time rescaling-- measuring time using different scales (ex. days, hours seconds). For the first iteration, Time2Vec utilizes a linear or non-periodic representation of the time vector and every other iteration is a function of the time vector representing the periodic feature. In the paper, it was shown that the sine function outperformed other non-linear functions such as sigmoid, tanh and ReLU in terms of accuracy and stability.

 <p align="center">
    <img src="https://user-images.githubusercontent.com/42477683/163737739-715d843d-fe11-4e79-9d45-a0f265324035.png" alt="alternate text">
 </p>

For our initial model we decided to use a a simple transformer model which contains a multiple attention heads, encoder and decoder. As mentioned in the `.ipynb` this model was inspired from original paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and Jan Schmitz's article 
[Stock predictions with state-of-the-art Transformer and Time Embeddings](https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6#:~:text=A%20Transformer%20is%20a%20neural,and%20Multi%2DHead%20Attention%20layer), 
to create our base model. Below you can see diagrams of One-head Attention archetecture, and Multi-head Attention archetecture we used for base model.

Please note that some parameters values might vary from the ones in the model to the fact that we changed some of them when we trained the model.

##### One-head Attention:

 <p align="center">
    <img src="https://user-images.githubusercontent.com/42477683/163856124-05358808-cb98-45f7-8ab5-2b6cbed24611.png" alt="alternate text">
 </p>


##### Multi-head Attention:

<p align="center">
    <img src="https://user-images.githubusercontent.com/42477683/163741346-54c95a8f-20ff-40dc-9cf7-dfa51167af61.png">
</Div>

### Data

We have dedicated the `transformer_prepros.ipynb` file to the handling of data; please consult the "Pre-Processing of the Data" section above.

Since the data is sorted by date, we cannot take random sequences from the data and set them as part of the validation and test sets. We leave 30% of the days in the stock history for testing and validation purposes. Then, the stock sequence is saved as an array of size [seq_len, 5], with the label being the next day's closing price.

Our training data consists of 12,573,778 data points, each with 128 days' worth of opening prices and 5 features: the opening and closing prices, daily high and low prices, and the volume traded. 

### Results

To get our results through training, we used the MSE loss. We chose this because it is common to linear regression applications and will give us a measure of how far the predictions are from the true values by using squared differences. There are no classes, we want to predict a particular value. We used the Adam optimizer because Adam is known to perform well across a variety of applications. 

We believe that our model should perform reasonably well on our test data set, since our model was built around the idea of Attention that works well for sequences. Unfortunately, the stock market is notoriously difficult to predict beyond the short term and so it is unclear how well our model would predict long-term stock prices given that world events can always have an impact.

Our next steps would also be to build the PyTorch version of the Transformer. 

### Ethical Consideration: 

It is important to understand that in order for one person to make money in the stock market and often times, another person typically loses money. Our model’s main purpose is to predict the future price of a stock. A successful stock market predictor can raise ethical issues if abused. This includes negatively impacting an excessive amount of buyers who take your loss or investing in companies that perform unethical business practices in orderto profit.  However, if used correctly, our model will help users develop a better overall understanding of factors affecting the stock market.

We are not obtaining our data from any insider sources and are only using historical stock data to train our model. In that sense our data is 100% safe and legal to use. Users must consider that because our model is not 100% accurate, the model cannot guarantee any profits. Users are at risk if using this model to perform real world trades. 

## Contributions
- Artur Azarskyy: researched articles and papers on topic, created the pre-processing workflow, created the StockDataset
 class, and sampler which could be used to train the model, writing of the documentation of the parts I did. I refactored the
 time embeding to be a torch model, I also created all of the other parts related to the basic model. Tested the model.
 to make sure that models runs.
- Jordan Tam: Researched articles and papers on topic, worked on Time2Vector, wrote about Ethical Consideration and the 
 other parts of that I worked on. Touched up README formatting and created model diagrams.
- Arsh Khan: Researched the topics of Transformers, Stock Prediction Models and the use of Transformers in Stock Prediction Models. Performed debugging on the data processing step. Due to Personal Issues was not able to contribute anything meaningful to the Project.
- Christopher Indris: Researched Transformers, Stock Prediction Models and Ethical Considerations. Commented and revised the preprocessing and model code. Revised this README and added the Results section.
