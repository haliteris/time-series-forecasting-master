from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
 
# convert series to supervised learning, shifting values to prepare future measurements.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN (ifany) values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def normalize(values):
    # normalize features
    normal = MinMaxScaler(feature_range=(0, 1))
    return normal

def fixnormal(normalvalues, values):
    #fit the normalized values.
    fixnormal = normalvalues.fit_transform(values)
    return fixnormal

def split_reshape(reframed,n_train):
   
    # split into train and test sets, n_train is determining the sample number to involve into training.
    values = reframed.values
    train = values[:n_train, :]
    test = values[n_train:, :]

    # split into input and outputs, 
    #training_t-period of t0 is dropped. Last 2 columns were indicated as inputs.
    train_X, train_y = train[:, :-3], train[:,-2:]
    print(train_X.shape, train_y.shape)
    #test_t-period of t0 is dropped. Last 2 columns were indicated as outputs.
    test_X, test_y = test[:, :-3], test[:, -2:]
    #check the shapes.
    print(test_X.shape, test_y.shape)
    print('output: ',train_y[1])

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, test_X, train_y, test_y