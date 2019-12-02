# Real Time Classification of Yoga Poses using SimpleHRNet

## Requirements:

- First clone and follow the instructions to install SimpleHRNet from this repository https://github.com/stefanopini/simple-HRNet
  - Be sure to you are able to run the demos listed in their directions
- After that is done clone the files from this repository and copy/paste them into the HRNet folder
- Run pip to install
```
pip install scikit-learn
```
- Run *MLP.py* to generate our MLP model
 
```
python MLP.py
```
- Run *app.py* to start the webcam application
   - Note you may have to pip install extra libraries if import errors come up
```
python app.py
```

## Additional files/folders
*get_keypoint_data.py* is used for extracting the raw kepoint data from the data located in the yogads folder
Run it if you want to see how the raw data is extracted
*keypoint_data.txt* is the output of this file

*gridsearch.py* are used for optimization and you may run it to find to see how we optimized our code

*sampleposes* just contains the images used in conjunction with the webcam


