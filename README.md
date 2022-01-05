# fgrm
Field goal range model using target zones for big data bowl

**When you retrain the model(ie generate new data), you may see different results from those presented**
The results should be similar, but they won't be exactly the same. All metrics are usually higher if the model is retrained.

python dataset_builder2.py - generates data
python dataset_splitter.py - creates training set and testing set
python k_nearest_neighbor.py - train and evaluate k nearest neighbors and generate data for decision tree
python white_box_interpreter.py - create decision tree
python eda.py - view statistics used to find statistics from the kaggle notebook
python field_goal_range_application.py - a text based wrapper for the FGRM presented that let's you experiment with different field goal attempts.
