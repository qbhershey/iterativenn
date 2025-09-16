# iterativenn
Iterative neural network code supporting several sparse RNN papers.

This architecture allows for iterative neural networks using a highly customizable block matrix approach following the approach developed by the research group under Randy Paffenroth.  This code library was used for several ICMLA papers by Quincy Hershey across a series of data sets and allows for easy comparisons with LSTM models.  The codebase operates on PyTorch Lightning, typically integrating Weights and Biases for results tracking.  The data sets integrated into this code include RL through Gymnasium using Minari supervised learning data sets within the Gymnasium Robotics library.  Additionally, a custom Random Anomaly data set was integrated with both discussed more thoroughly below.  The characterstics and structure of the code base are outlined in the sections that follow.

Runner.py files are the primary python files used to execute training or testing runs.
* Runner_AD.py used to generate and log batches of runs using conf files for the Adroit Door data set
* Runner_AH_Uniform.py was used to generate and log batches of randomly specified networks on the Adroit Hammer data set
* Runner_AH.py used to generate and log batches of runs using conf files for the Adroit Hammer data set
* Runner_AHV.py used to generate and log batches of runs with varied sparsity on the Adroit Hammer data set
* Runner_FK.py used to generate and log batches of runs using conf files for the Franka Kitchen data set
* Runner_RA_Uniform.py was used to generate and log batches of randomly specified networks on the Random Anomaly data set
* Runner_RA.py used to generate and log batches of runs using conf files for the Random Anomaly data set
* Score_AD.py is a utility file used to score average reward for models trained on minari data sets using the associated Gymansium Robotics environment
* minari_reward.py is a utility file used to calculate store and retrieve average reward for various minari data sets

Config files (.yaml) are typically housed within the conf folder with and are used to compile the specifications for each run.  The conf structure may be modified but currrently combines a conf file from each of the following subdirectories into the main config.yaml file within the conf folder.
* base establishes parameters such as number of epochs, cuda, logging, etc
* data specifies the data set
* logger determines how and where to log results
* model is used to specify model type, characteristics, initializations, etc

The Data folder includes several subfolders used to house the DataFactory.py files which construct the data sets for training and testing.  Several of the data files used to run code in these folders are missing owing to the size limits in github.  Notably, the RnadomAnomaly data folder is missing two files compiled using this process (RandomAnomaly_Train_Combined30.parquet.gzip and RandomAnomaly_Train_Combined58.parquet.gzip) which are too large to host here as they combine for 1.71GB but are available upon request.  Several of the minari data files housed in the .minari folder are also missing as they were too large to upload (above the 100MB size limit), those are listed below.  Missing files are available upon request.
* AdroitDoor from Gynasium's Minari which employs an environment from Gynamsium Robotics.
    - missing .minari\datasets\door-cloned-v1\data\main_data.hdf5
    - missing .minari\datasets\door-expert-v1\data\main_data.hdf5
* AdroitHammer from Gynasium's Minari which employs an environment from Gynamsium Robotics
    - missing .minari\datasets\hammer-cloned-v1\data\main_data.hdf5
    - missing .minari\datasets\hammer-expert-v1\data\main_data.hdf5
* FrankaKitchen from Gynasium's Minari which employs an environment from Gynamsium Robotics
    - missing .minari\datasets\kitchen-mixed-v1\data\main_data.hdf5
    - missing .minari\datasets\kitchen-partial-v1\data\main_data.hdf5
* RandomAnomaly which uses data from https://github.com/qbhershey/RandomAnomaly for an anomaly detection task based on MNIST.  Details regarding the data set are available at that repository.
    - missing Data\RandomAnomaly\RandomAnomaly_Train_Combined30.parquet.gzip
    - missing Data\RandomAnomaly\RandomAnomaly_Train_Combined58.parquet.gzip

The iterativenn folder houses the engine used for running block function based networks such as the sparse RNNs as well as comparison LSTMs.

