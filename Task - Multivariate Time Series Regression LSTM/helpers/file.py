import pandas as pd

#Prepare (clear any space or tabs, seperate them with commas) and load the data with new headers.
def prepare_dataset():
    header= ['t', 'coppm','ethyleneppm','sensor1','sensor2','sensor3','sensor4','sensor5','sensor6',
          'sensor7','sensor8','sensor9','sensor10','sensor11', 'sensor12', 'sensor13', 'sensor14', 'sensor15',
          'sensor16']
    dataset=pd.read_csv(r'C:\Users\HalitPC\Downloads\ml-task 1\st-task\workspace\st-task\datasets\ethylene_CO.txt',
                    skiprows=[0],names=header, sep=r"\s+")
    print('Dataset is succesfully loaded. Summary of the dataset is shown below:  ' )
    return dataset