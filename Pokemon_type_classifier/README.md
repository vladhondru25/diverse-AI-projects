# Pokemon type classifier

The aim of the mini-project was to classify a Pokemon in one of the 18 types: Normal, Fire, Water, Grass, Flying, Fighting, Poison, Electric, Ground, Rock, Psychic, Ice, Bug, Ghost, Steel, Dragon, Dark and Fairy. This is a challenge from Kaggle: <https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types>. 

## How to use?
Simply run the cells in the Jupyter Notebook in order on the Kaggle website in the respective challenge. It is recomended to use the GPU for faster training.


## Features of the training algorithm
A csv file which holds the Pokemons names and their types is read and the data is processed using Pandas. A dataset is created using Pytorch utils library. The type is encoded using an integer from 0 to 17 using sklearn. Data augmentation is used on the data (horizontal flips and rotations between -45deg and 45deg). Three architectures were tried (all of them are found, 2 of them commented out), started from a more complex one and then reducing the capability due to overfitting on the data. The resulting architecture is composed of 4 convolutional layers with batch normalisation, dropout and rectified activation function, after the first three max pooling is applied, and finally a fully connected layer. 

The model was trained for 200 epochs, with Adam optimizer. The final accuracy on the testing dataset was 90.12%. The training and testing losses are plotted to verify if overfit is taking place. NOTE: Validation and testing dataset is the same, however, the validation dataset is not used in learning not in hyperparameter choice. Ten results are plotted in the end, with the image, predicted type and actual type, as below. 

![alt text](https://github.com/vladhondru25/diverse-AI-projects/blob/master/./Pokemon_type_classifier/result1.png?raw=true)

![alt text](https://github.com/vladhondru25/diverse-AI-projects/blob/master/./Pokemon_type_classifier/result2.png?raw=true)

![alt text](https://github.com/vladhondru25/diverse-AI-projects/blob/master/./Pokemon_type_classifier/result3.png?raw=true)
