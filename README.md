# iterativenn
Iterative neural network code supporting several sparse RNN papers.

This architecture allows for iterative neural networks using a highly customizable block matrix approach following the approach developed by the research group under Randy Paffenroth.  This code library was used for several ICMLA papers by Quincy Hershey across a series of data sets and allows for easy comparisons with LSTM models.  The codebase operates on PyTorch Lightning, typically integrating Weights and Biases for results tracking.  The data sets integrated into this code include RL through Gymnasium using Minari supervised learning data sets within the Gymnasium Robotics library.  Additionally, a custom Random Anomaly data set was integrated with both discussed more thoroughly below.  The characterstics and structure of the code base are outlined in the sections that follow.

Runner.py files are the primary python files used to execute training or testing runs.
* Runner_AD.py
* Runner_AH_Uniform.py
* Runner_AH.py
* Runner_AHV.py
* Runner_FK.py
* Runner_RA_Uniform.py
* Runner_RA.py
* Score_AD.py
* minari_reward.py

Config files (.yaml) are typically housed within the conf folder with and are used to compile the specifications for each run.  The conf structure may be modified but currrently combines a conf file from each of the following subdirectories into the main config.yaml file within the conf folder.
* base establishes parameters such as number of epochs, cuda (True/False), logging, etc
* data specifies the data set
* logger determines how and where to log results
* model is used to specify model type, characteristics, initializations, etc

The Data folder includes several subfolders used to house the DataFactory.py files which construct the data sets for training and testing.
* AdroitDoor from Gynasium's Minari which employs an environment from Gynamsium Robotics
* AdroitHammer from Gynasium's Minari which employs an environment from Gynamsium Robotics
* FrankaKitchen from Gynasium's Minari which employs an environment from Gynamsium Robotics
* RandomAnomaly which uses data from https://github.com/qbhershey/RandomAnomaly for an anomaly detection task based on MNIST.  Notably, this data folder is missing two files compiled using this process (RandomAnomaly_Train_Combined30.parquet.gzip and RandomAnomaly_Train_Combined58.parquet.gzip) which are too large to host here as they combine for 1.71GB but are available upon request.  Details regarding the data set are available at that repository.





