Here we include description of the directories and files for this method.

**dataset**: <br />
This directory includes the "read_data.py" code for parsing the ECHR dataset and create an output csv file with the following columns: <br />
1) id -- a running number used for the method.
2) name -- the name of the judgment text file (from "00.txt" to "42.txt").
3) argument -- a premise in the text.

**run_kpa_scripts**: <br />
This directory includes the "run_keypoint_analysis_script.py" which can run the debater in one of the two pipelines as mentioned in the paper. <br />
1) Run on all texts together and create one result file <br />
2) Run on each text separately and create result file for each, then merge the results files to one output file. <br />

Our default behavior is the second pipeline. The input files for each text as well the result files are included in the folder, so you don't have to run debater.

**results**: <br />
This directory contains the results xlsx <br />
1) "results_all_judgments_together.xlsx" -- result for the first pipeline
2) "results_judgments_separately.xlsx" -- result for the second pipeline

