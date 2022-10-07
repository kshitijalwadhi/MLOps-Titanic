# MLOps Titanic Challenge

In this repository, we have the code for two tasks:
* Local Experimentation
* Deploying on Vertex AI

The file structure of this repository is as follows:

```go
+--- titanic-package // contains files used for deploying on vertex AI
    +--- ..
    +--- .. 
+--- model.pkl // model file generated from execution of script.py
+--- script.py // the solution to the coding task
+--- test.ipynb // the original notebook where experimentation was done 
+--- train.csv // the data
```

To install the dependencies, please execute the following command in your terminal:
``` 
pip install -r requirements.txt
```

Note that the file `script.py` assumes that the data `train.csv` is present in the same location as that of the code for training the model. The file was built for making the classes and not for testing purposes. The main function in this file just calls upon the classes implemented to verify the working.