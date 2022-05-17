# Paper: Key Point Analysis via Contrastive Learning and Extractive Argument Summarization

This is the code for the paper *Key Point Analysis via Contrastive Learning and Extractive Argument Summarization*.

Milad Alshomary, Timon Gurcke, Shahbaz Syed, Philipp Heinrich, Maximilian Spliethöver, Philipp Cimiano, Martin Potthast, Henning Wachsmuth


# Code
Only some sections were edited to try to execute the model
*Note* Some files are missing/Model

## For our Generation Track:

For key-point generation the following experiment notebooks should be executed:

- The `experiment-data-prep-for-track-2.ipynb` notebook contains the code that generate argumentative quality scores needed to run the ArgPageRank
- The `experiment-page-rank.ipynb` notebook contains the code to generate the key-points using the ArgPageRank
- The `experiment-evaluation.ipynb` notebook contains the code to perform the final evaluation

There is no need to run the first notebook as i ran and saved the pkl 
For the second notebook some data is missing. I tried to manually recover it but still damage the performance and cause some errors
### Additional dependencies
For SmatchToPr PageRank we need rouge , rouge_score as well as sentence_transformers and torch that can be installed using pip 


