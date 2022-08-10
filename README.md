**Abstract**:

Key Point Analysis(KPA) is a relatively new task in NLP that combines summarization and classification by extracting argumentative key points (KPs) 
for a topic from a collection of texts and categorizing their closeness to the different arguments.

In our work, we focus on the legal domain and develop methods that identify and extract KPs from premises derived from texts of judgments. 
The first method is an adaptation to an existing state-of-the-art method, and the two others are new methods that we developed from scratch. 
We present our methods and examples of their outputs, as well a comparison between them. 
The full evaluation of our results is done in the matching task -- match between the generated KPs to arguments (premises). 

**KP Extraction Task**:

Given a collection of arguments towards a certain topic, the goal is to generate KP-based summary. 
KPs should be concise, non-redundant, and capture the most important points of a topic. 
Ideally, they should summarize the input data at the appropriate granularity -- general enough to match a significant portion of the arguments, 
yet informative enough to make a useful summary. Then, in the matching task, the goal is to compute the confidence score between arguments to the extracted KPs. 

In our work, the arguments are premises from different texts of judgments, and we apply this task on them. 

**The Dataset**:

We use the European Court of Human Rights (ECHR) dataset to extract KPs in all of our methods. It consists of 42 human-annotated judgements. 
The corpus is annotated in terms of premises and conclusion. Overall, 1951 premises and 743 conclusions. 
We consider the premises as arguments, and extract the KPs from them. 

Example of an argument: 
``The Commission considers that this indicates an issue falling within the scope of freedom of expression''.

The dataset input json file is in the ``Dataset'' directory.

**The Methods**:
All the code and the output files are in the directories of the different methods:
1) KP Candidate Extraction and Selection Using IBM
2) Clustering and Summarization
3) PageRank
