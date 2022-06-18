![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/logo2.png)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

 <tr><td colspan="4"> <a href="#Get Started"> Get Started </a></td></tr> | <tr><td colspan="4"> <a href="#Dataset Usages"> Dataset Usages </a></td></tr> | <tr><td colspan="4"> <a href="#Pipeline Usages"> Pipeline Usages </a></td></tr> | <tr><td colspan="4"> <a href="#Evaluation & Visualization Toolkit"> Evaluation & Visualization Toolkit </a></td></tr> | <tr><td colspan="4"> <a href="#Benchmarks"> Benchmarks </a></td></tr> | <tr><td colspan="4"> <a href="#Acknowledgement"> Acknowledgement </a></td></tr>

 ## Get Started
 
 CGLB needs the following packages to be installed:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1
* dgllife==0.2.6
 
 ## Dataset Usages
 
 ### Importing the Datasets
 For importing the N-CGL datasets, use the following command in python:
 
 ```
 from CGLB.NCGL.utils import NodeLevelDataset
 dataset = NodeLevelDataset('data_name')
```
 
 where the 'data_name' should be replaced by the name of the selected dataset, which are:
 
 ```
 'CoraFull-CL'
 'Arxiv-CL'
 'Reddit-CL'
 'Products-CL'
 ```
 
 For importing the G-CGL datasets, use the following command in python:
 ```
 from CGLB.GCGL.utils import GraphLevelDataset
 dataset = GraphLevelDataset('data_name')
 ```
 
 where the 'data_name' selections are:
 ```
 'SIDER-tIL'
 'Tox21-tIL'
 'Aromaticity-CL'
 ```
 
 
 ### Data Structures and Formats
 
 An instance of the ```NodeLevelDataset``` has multiple attributes and methods. ```dataset.d_data```, ```dataset.n_cls```, and ```dataset.n_nodes``` denotes the number of dimensions of the node features, number of classes of the corresponding dataset, and the number of nodes in the corresponding dataset, respectively. ```dataset.graph``` is the entire graph of the corresponding dataset without being divided. The data type of ```dataset.graph``` is the [DGL graph](https://docs.dgl.ai/tutorials/blitz/index.html). ```dataset.labels``` contains the label of every node of ```dataset.graph```. ```dataset.tr_va_te_split``` is a dictionary containing the splitting of each class for training, validation, and test.
 
 ```dataset.get_graph()``` is a most useful method which returns a subgraph of ```dataset.graph``` (for constructing the continual learning task sequence) with the given classes or node indices to retain. There are three keyword arguments for ```dataset.get_graph()``` including ```classes_to_retain```, ```node_ids```, and ```remove_edges```. ```classes_to_retain``` is used to specify which classes of the ```dataset.graph``` to retain in the returned subgraph. ```node_ids``` is used to specify specific nodes to retain. If both ```classes_to_retain``` and ```node_ids``` are specified, their corresponding subgraphs will be separately generated and then combined into one graph. ```remove_edges``` is specifically for removing the edges of the subgraph generated from the specified ```node_ids```. One possible usage of this argument is to select and output the nodes to store for the memory based baseline ER-GNN.
 
 The usage of the G-CGL dataset resembles the classic continual learning. An instance of ```GraphLevelDataset``` consists of four components. ```dataset.dataset``` is the original dataset before being divided. ```train_set, val_set, test_set``` are the splittings for training, validation, and test. For the task-IL datasets ```SIDER-tIL``` and ```Tox21-tIL```, the original datasets are multi-label ones, and the splitting is simply splitting the dataset into three parts according the given ratios. While for the ```Aromaticity-CL``` whose original dataset is a multi-class one, the splitting is also done for each class like the N-CGL datasets. As a result, for ```Aromaticity-CL```, each of ```train_set```, ```val_set```, and ```test_set``` is a list containing the splitting of each class. 
 
 ## Pipeline Usages
 
 We provide pipelines for training and evaluating models with both N-CGL and G-CGL tasks under both task-IL and class-IL scenarios. In the following, we provide several examples to demonstrate the usage of the pipelines.
 ### N-CGL
 Below is the example to run the 'Bare model' baseline with GCN backbone on the Arxiv-CL dataset under the task-IL scenario. 
 
 For both N-CGL and G-CGL experiments, the starting point is the ```train.py``` file, and the different configurations are assigned through the keyword arguments of the Argparse module. For example, to run the N-CGL experiments without inter-task edge under the task-IL scenario, the following code is to be used.
 
 ```
 python train.py --dataset $Arxiv-CL \
        --method $Bare \
        --basemodel $GCN \
        --gpu 0 \
        --clsIL False
 ```
 By specifying the ```--clsIL``` to be ```False```, the experiments are configured under the task-IL scenario. 
 ### G-CGL
 Below is an example for running the 'Bare model' baseline with GCN backbone on the SIDER-tIL dataset under the task-IL scenario. 
 ```
 python train.py --dataset $SIDER-tIL \
        --method $Bare \
        --basemodel $GCN \
        --gpu 0 \
        --clsIL False
 ```
 
 
 ## Evaluation & Visualization Toolkit
 We provide three protocols to evaluate the obtained results as follows. With out pipeline, the results are uniformly stored in the form of performance matrix, which can be directly fed into our evaluation toolkit.
 
 ### 1. Visualization of the Performance Matrix
 
 This is the most thorough evaluation of a continual learning model since it shows the performance change of each task along the learning process on the entire task sequence. Suppose an experiment result is stored via the path ``` "result_path" ```, the generation of the visualization could be obtained by the following code. Note that the path should be quoted in ``` " " ``` instead of ``` ' ' ```, since ``` ' ' ``` may exist in the file name of the experimental result.
 ```
 from CGLB.NCGL.visualize import show_performance_matrices
 show_performance_matrices("result_path")
 ```
 
 ### 2. Learning Curve
 
 This shows the curve of the average performance (AP). It contains less information than the performance matrix but can demonstrate the learning dynamics in a more direct and compact way. Suppose an experiment result is stored via the path ```result_path```, the learning curve could be obtained by the following code.
 ```
 from CGLB.NCGL.visualize import show_learning_curve
 show_learning_curve("result_path")
 ```
 
 ### 3. Final AP and Final AF
 Final AP and AF refers to the AP and AF after learning the entire task sequence and is the most compact way to show the performance of a model. Suppose an experiment result is stored via the path ```result_path```, the final AP and AF could be obtained by the following code.
 ```
 from CGLB.NCGL.visualize import shown_final_APAF
 shown_final_APAF("result_path")
 ```
 The outputs with standard deviation are in LaTex form for making it easy to be copied and pasted into a LaTex table.
 
 
 ## Benchmarks
 This section shows our currently obtained results from different baselines. This section will keeps being updated to show state-of-the-art results.
 ### N-CGL under Task-IL
 
 ### N-CGL under Class-IL
 
 ### G-CGL under Task-IL
 
 ### G-CGL under Class-IL
 
 ## Acknowledgement
 The construction of CGLB also benefits from existing repositories on both continual learning and continual graph learning. Specifically, the construction of the pipeline for training the continual learning models learns from both [GEM](https://github.com/facebookresearch/GradientEpisodicMemory) and [TWP](https://github.com/hhliu79/TWP). The implementations of the implementations of EWC, GEM learn from [GEM](https://github.com/facebookresearch/GradientEpisodicMemory). The implementations of MAS, Lwf, TWP learn from [MAS](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses) and [TWP](https://github.com/hhliu79/TWP). The implementation of TWP is adapted from [TWP](https://github.com/hhliu79/TWP). The construction of the datasets also benefits from several existing databases and libraries. The construction of the N-CGL datasets uses the datasets and tools from OGB and DGL. The construction of the G-CGL datasets uses the datasets and tools from [DGL](https://docs.dgl.ai/) and [DGL-Lifesci](https://lifesci.dgl.ai/api/data.html).
We sincerely thank the authors of these works for sharing their code and helping developing the community.
