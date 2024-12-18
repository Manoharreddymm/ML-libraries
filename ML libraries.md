# `                      `**ML libraries**
Pandas:

The pandas works in the datasets.

Pandas are used for analysing, cleaning, manipulating the data present in the dataset.

It allows us to analyse the big data and make conclusion from statistical theories in pandas.

Pandas Series- it is a column in the table.

In the series we don’t have any column name and has no identification in the series, that’s why we are using the dataframe.

DataFrame- the dataframe is a 2 dimensional data structure (rows and columns).

For reading csv file - pd.read\_csv(‘iris.csv’)

Some functions in the pandas:

- Head()-gives top 5 rows
- to\_string-returns entire dataframe
- select\_dtypes(include=[‘int64,float64’] or [‘object’])
- describe() – describes about the data
- unique()-gives unique values present in the column
- value\_counts()-gives the count of every unique value in column
- iloc- index location
- loc –loacation with the name
- dropna() and fillna() – drops or filling the empty values in the column
- duplicate() and drop\_duplicates() – finding the duplicate values and dropping them.

**Matplotlib and Seaborn:**

These libraries are used for visualization purpose and check the relation between the columns in visualized manner.

The matplot ia a low level graph plotting library that serves as a visualization utility.

Pyplot most used in the matplotlib utilities lies under the pyplot.

Plotting functions:

- plt.plot()
- plt.scatter()
- plt.bar()
- plt.histplot()
- plt.pie()
- plt.boxplot()

customization functions:

- plt.title()
- plt.xlabel()
- plt.ylabel()
- plt.legend()
- plt.grid()

Relational plot:

Gives the relation between 2 columns

The basic requirement of these plots are 2 contineous numeric columns

1. lineplot()
1. scatterplot()
1. relplot()

Distribution plot:

These are used for single numeric column.

1. Histplot()
1. Distplot()- tells us weather the data has outliers or not

Categorical plot:

1. Countplot()
1. Barplot()


**Scikit learn:**

The scikit learn mainly used in the machine learning algorithms. The scikit learn functions are already written in the python and open source any one can modify it.

It provides the wide range of tools for data analysis, including algorithms for classification, regrssson, preprocessing, metrics, clustering, model evaluation .

1\.sklearn.databases

`	`In this library there are some datasets that are already present in the sklearn. The databases can be regression data and classification data.

Some of the databases:

- load\_iris()
- load\_digits() 
- load\_wine()
- load\_breast\_cancer()
- load\_diabeties()

2\.sklearn.model\_selection

`	`For model building we need to divide the data into test and train datasets. To divide it we use the model\_selection

Used for splitting databases, cross\_validation and hyper parameter tuning.

- train\_test\_split()- splitting the data
- cross\_val\_score()-performance of cross validation
- GridSearchCV()-hyperparameter tuning
- KFold()- cross validation folds

3\.sklearn.preprocessing

`	`Preprocessing techniques are applied on the dataset before building the model. Preprocessing techniques like scaling, normalization, encoding and feature selection.

- StandardSelection()- standard features by removing mean and scale to unit varience.
- MinMaxScaler()- scale features in range 0 and 1
- LabelEncoder()- encoding for categorical features

4\.sklearn.linear\_model

`	`Linear models for classification and regression

- LinearRegression()
- LogisticRegression()
- Lasso() and ridge() – regulization technique for linear regression
- SGDClassifier(),SGDRegressor()- stochastic gradient decent to large scale models

5\.sklearn.ensemble

`	`Ensemble methods combine multiple models to improve performance.

(Bagging and Boosting)

- RandomForestClassifier(),RandomForestRegression()
- AdaBoostClassifier(),AdaBoostRegression()- adaptive boosting method

6\.sklearn.svm

`	`Support vector machine for classification and regression

- SVC() and SVR()
- LinearSVC() and LinearSVR()

7\.sklearn.tree

`	`Decision tree algorithms for classification and regression

- DecisionTreeClassification()
- DecisionTreeregression()

8\.sklearn.naive\_bayes

- GaussianNB()- classifier for continuous data
- MultinomialNB()- classifier for descrete data
- BernoulliNB()- classifier for binary data

9\.sklearn.decomposition

`	`Used for the dimensionality reduction purpose

- PCA()
- TruncatedSVD()- dimensionality reduction for sparse matrix

10\.sklearn.metrics

Provides functions to evaluate the performance of models

- Accuracy\_score(),precision\_score(),recall\_score(),f1score()- metrics for classifier
- Mean\_square\_error(),r2\_score()-metrics for regression

**KERAS:**

`	`The keras is implemented for the humans. Keras is broadly adopted for industries among research.

It supports multiple backend engine. The high level neural network API run on the top of tensorflow and theano.

Keras allows us to build ,train and deploying deep learning models with minimal code.

The models present in the keras:

- Sequential model
- Functional model

Sequential model:

`	`Linear stack of layers present in the neural nodel like first the input layer then hidden layers and then output. In this model we cannot give the input in the middle.

from keras.model import sequential

from keras.layer import Dense,Activation

model=sequential()

model.add(Dense(unit=64,input\_dim=100))

model.add(Activation(‘relu’))

model.add(Dense(unit=10))

model.add(Activation(‘softmax’))

Functional model:

`	`The keras function API is used for defining complex models such as multi-output layers, direct acyclic graph.

3 unique aspects of the keras function API:

- Defining the input
- Connecting the layers 
- Creating the model

from keras.layers import Input,Dense,Conactenate

from keras.models import model

ip1=Input(shape(100,))

ip2=Input(shape(50,))

h1=Dense(64,activation=’relu’)(ip1)

h2=Dense(32,activation=’relu’)(ip2)

merge=Conactenate([h1,h2])

output=Dense(10,activation=’softmax’)(merge)

model=Model(inputs=[ip1,ip2],outputs=output)

keras has a number of predefined layes core layer, convolutional layer, pooling layer, local-connect layer, recurrent layer.

**Xgboost:**

`	`It is a scalable, distributed gradient decent tree machine learning library. It is a ensemble learning method that combines the prediction of multiple weak models to produce the strong prediction.

The Xgboost has the ability to handle large datasets.

The Xgboost itself takes care about the null values. It automatically learns how to handle missing values in the data.

**LightGDM:**

`	`It is a gradient boosting framework that uses tree-based learning algorithms and is designed to be fast and efficient, particularly on large datasets.

It contains nodes represent features splits and leaf node contains predictions. Constructed recursively in a leaf wise manner.

It can handle the categorical data.

Light Gradient Boosting Machine, is an advanced machine learning algorithm designed to improve the speed and efficiency of gradient boosting. Developed by Microsoft, it excels in handling large datasets and high-dimensional feature spaces. LightGBM uses a unique approach known as "gradient-based one-side sampling," which selects a subset of data for training, and "exclusive feature bundling," which reduces the dimensionality of data without sacrificing performance. These features enable faster training times and lower memory usage compared to traditional gradient boosting algorithms, making LightGBM a popular choice for tasks like classification, regression, and ranking.

**TENSORFLOW:**

`	`TensorFlow is an open-source software library that helps developers build machine learning applications.

Tensorflow can train and run deep neural network for image recognization, handwritten digit classification, RNN and NLP.

Tensorflow is a 2D array that follows the flow of the structure. All values present in a tensor are same datatype.

Models can be built using the Sequential or Functional API, with layers such as Dense defining neural network architecture. Before training, models must be compiled with an optimizer, a loss function and evaluation metrics like accuracy. Data pipelines can be created using the tf.data API, which allows efficient batching, shuffling, and prefetching of data.

Training a model is done using the fit() method, which accepts datasets and can include validation data for monitoring performance. Once trained, the model can be evaluated with evaluate() or used for inference with predict(). Models can also be saved and reloaded using model.save() and tf.keras.models.load\_model() for reuse in production. TensorFlow supports custom training loops with GradientTape for more granular control over training steps, while distributed training across multiple devices can be achieved using tf.distribute.Strategy. TensorFlow Lite enables deploying models on mobile and embedded systems, and TensorFlow Extended (TFX) facilitates end-to-end machine learning pipelines for production environments.

Visualization of training progress and metrics is made possible by TensorBoard, TensorFlow’s built-in tool, which can be integrated through a callback during training. TensorFlow also includes debugging features via tf.debugging for validating and inspecting models. With its extensive ecosystem and flexibility, TensorFlow is a powerful tool for developing and deploying machine learning models at scale.

**PYTORCH:**

`	`PyTorch is an open-source deep learning library that provides flexibility and speed for building machine learning models. Its core feature is the dynamic computation graph, which allows for flexible network architecture design and easy debugging. Tensors, which are multi-dimensional arrays similar to NumPy arrays but with GPU support, form the foundation of PyTorch. The autograd engine enables automatic differentiation by tracking operations on tensors with requires\_grad=True, allowing for easy gradient computation during backpropagation. 

PyTorch's nn module simplifies building neural networks by providing pre-built layers and activation functions, while optimizers such as torch.optim.SGD or Adam are used for updating model parameters. 

The DataLoader and Dataset classes help manage large datasets efficiently, enabling mini-batch training. Additionally,

` `PyTorch supports GPU acceleration via CUDA, making it easy to run computations on a GPU with to(device) calls. Pre-trained models for transfer learning are available in libraries like torchvision, which contains models such as ResNet or VGG.

` `PyTorch also provides easy-to-use tools for saving, loading, and deploying models, with TorchScript enabling the conversion of models into optimized production formats. Its flexibility, combined with a user-friendly API, makes PyTorch a popular choice for research and industry applications.
