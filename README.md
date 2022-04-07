# CSC413 Stock Prediction project

Our project consists of mainly two parts the preprocessing of the data located in `transformer_prepros.ipynb`,
and the ML models would be located in `transformer_model.ipynb`.

## Pre-processing of the data:

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




## Executing the Model:

### Prep:

First we get the pre-processed zipped files from the Google Drive. Since we want to use PyTorch we need to define a custom 
Dataset class which given the name of the `.hdf5` file will make access to both `data` and `lables` directories of the file
available. We also introduce transformation function which transforms the numpy values into tensors, and we also have a
transformation which will help us to ensure that labels are of right shape if we decide to use RandomBatchSampler. I also
added a small benchmarking code which shows that the data is correctly loaded and benchmarking the spead of loading of 
both loaders

### Model:

## Contributions:
- Artur Azarskyy: researched articles and papers on topic, created the pre-processing workflow, created the StockDataset
 class, and sampler which could be used to train the model, writing of the documentation of the parts I did.
- 
[//]: # (TODO: everyone please add your names and contributions here )
