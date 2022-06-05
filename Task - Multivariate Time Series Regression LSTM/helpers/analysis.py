import seaborn as sns
import matplotlib.pyplot as plt

#dataset may have NaN values. Check the NaN values.
def check_nan(dataset):
    print(dataset.isna().any())  

#Describing the dataset in means of:
#count, mean, standard deviation, min. value, max.value, lower, median, upper percentages for per feature.
def description_stats(dataset):
    print(dataset.describe())

#Feature correlation heatmap to determine the correlation coefficients between variables.
def correlation(dataset,figx, figy, maxv, square_bool):
    cor_metrics= dataset.corr()
    fig= plt.figure(figsize=(figx,figy))
    sns.heatmap(cor_metrics, vmax=maxv, square= square_bool)
    plt.show()
    print('Correlation metrics: \n',cor_metrics)

#Check number of the output unique values.
def dist_count(dataset,column):        
    if column=='co':
        print('CO Measurements', '\n')
        print('Unique PPM Value/Count','\n')
        item_counts_co= dataset['coppm'].value_counts()
        print(item_counts_co, '\n')
        sns.countplot(x=dataset.coppm)
        plt.title('Unique Count')
        
    elif column=='ethylene':
        print('Ethylene Measurements', '\n')
        print('Unique PPM Value/Count','\n')
        item_counts_ethylene= dataset['ethyleneppm'].value_counts()
        print(item_counts_ethylene, '\n')
        sns.countplot(x=dataset.ethyleneppm)
        plt.title('Unique Count')

#Plotting the values of each feature over time.
def graph_each(dataset):   
    co=dataset['coppm']
    x=dataset['t']
    fig, ax = plt.subplots(figsize=(15,2))
    ax.plot(x, co, 'tab:orange')
    ax.set_title('Distribution for : CO ppm')
    
    ethylene=dataset['ethyleneppm']
    fig, ax = plt.subplots(figsize=(15,2))
    ax.plot(x, ethylene,'tab:green')
    ax.set_title('Distribution for : Ethylene ppm')
    
    for i in range(1,17):
        param= 'sensor'+str(i)
        a=dataset[param]
        x=dataset['t']
        fig, ax = plt.subplots(figsize=(15,2))
        ax.plot(x, a)
        ax.set_title('Distribution for :'+param)