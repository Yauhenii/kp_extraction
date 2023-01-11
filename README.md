# SmatchtoPR

Link to the model: https://drive.google.com/drive/folders/1bScJ9ObvBxfQEDjhWBcrd-zoSDRwgnA9?usp=share_link

In order to run the SmatchtoPr for Kp extraction. We need to install the [IBM Debater_api](https://early-access-program.debater.res.ibm.com/) that is used to calculate the argument quality using the following steps

### Step 1: Install debater API/spacy
#### windows
install the sdk and unzip it. To install the api and its dependencies in a new env 
```bash
#Create a conda env:
conda create --name <name> python=3.7
#Activate env:
conda activate <name>
```
In the root folder of the api run 
```bash
pip install .
```
Note: spacy 2.2.1 is required for the api to be functional (v3 doesn't contain some modules the api uses). spaCy 2.2.1 is compatible only with en_core_web_sm  2.2.0. Therefore it is required to install it from the source 
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
```
To understand some basics of the Debater_api. Have a look at this [Tutorial](https://github.com/IBM/debater-eap-tutorial)
### Step2 : Running Kp-extraction-track related method
The SmatchtoPr Method for key generation is implemented under https://github.com/webis-de/argmining-21-keypoint-analysis-sharedtask-code which is the implementation of the paper  [Key Point Analysis via Contrastive Learning and Extractive Argument Summarization](https://webis.de/downloads/publications/papers/alshomary_2021b.pdf) .  The KPa_2021_shared_task should be cloned as a submodule.

The Computation time to calculate the score arg/topic on the training set is huge (took us 460 min+ runtime the pkl file for training that contains arg/scor to topic is  provided ).* Do not uncomment the lines commented as it will overwrite the variables (a backup is provided)




