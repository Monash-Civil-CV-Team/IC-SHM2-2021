# IC-SHM2-2021
Earthquakes accounted for around 1.87 million deaths in the past century with building collapses identified as the major fatal cause. In a quick response to earthquakes, condition assessment of damaged buildings is a critical task to provide timely rescue of lives, informed decision making and mitigation measures to secondary hazards. Manual inspection can be laborious, causing delay of subsequent actions, and sometime hazardous, resulting in worker casualties due to the secondary effects like aftershocks.  For example, a devastating tragedy of  44 fatalities including rescuers and journalists caused by the aftershock in Van, Turkey in 2011 was reported. These concerns are calling for automatic alternatives to conduct post-earthquake building assessment.
# Data
![image](https://user-images.githubusercontent.com/77284145/188786135-8473ce7d-7038-4338-8d62-6cdd367532e3.png)
The project’s overall aim is to automate UAV-based health estimation of post-earthquake buildings, which encompasses 5 pixel-wise sub-objectives. Included are to identify structural components, detect damage regions (spall, crack, and exposed rebar), and assess damage states. Two challenges were identified, arising from the varied environments and sizes of tracking targets. First, the multiple classes to be predicted are highly correlated with each other. Some components of key structural importance such as column and beam are more likely to be damaged in seismic events. Particularly, the damage state estimation task, which is the ultimate goal of condition assessment, is dependent on the presence of defects. For instance, rebar only exposes at the premise of spalling and usually indicate the severe damage state of the region (Fig. 1).  Second, the tremendously varied target sizes among sub-tasks poses another challenge of prediction bias for data-driven algorithms. The structural component may be identified much more easily from remote sight while the damages possessing smaller sizes such as crack and rebar exposure can only be visibly captured when the camera is closely present. 
![image](https://user-images.githubusercontent.com/77284145/188786336-3bd09bf7-7f6d-4fd5-88b6-196a12cf1007.png)
The provided dataset consists of 3,805 annotated pictures with size of 1,080×1,920 captured from damaged buildings by UAVs at different heights and views. The 80% of original acquisitions was randomly extracted to form the training set and the remaining was used for model validations. Two dataset preparation modes were adopted in this study (Fig. 3) to tackle the sample imbalance problem for the sub-tasks. In plain mode, images of the original size were directly rescaled to 512×512 without any further manipulation. In filtering mode, where most information can be retained, each image was cropped to 8 patches with size of 512×512. For the small object segmentation tasks (spall and crack), subject to the non-equilibrium distribution, extra efforts were applied to filter out the overwhelming background. Cropped patches with positive sample (spall and crack) less than the specified threshold value were discarded. Sample distributions were considerably balanced after oversampling in the filtering mode, as shown in Table 1. The plain dataset was fed into the model with component and damage status estimation as the main tasks while the three balanced datasets of spall and crack were used to optimize the corresponding U-Net.  
# Network Architecture
![image](https://user-images.githubusercontent.com/77284145/188786021-5c2a0d59-2f1b-40ee-9229-035b3fbb5784.png)
# Enviroment
Please run pip install -r requirements.txt 
# Procedure
—__1.Data Preparation__  
Please download the data through this link [Original data](http://www.schm.org.cn/#/IPC-SHM,2020/dataDownload) and put the data at the root of the project.
the image process code is in folder "data processing" are used to precess the data. Please binary the mask images first, then cut and crop the images and masks.
If you try to use your own data, please use following format  
 >--Data  
 >> --train  
 >>> --image  
 >>> --mask  
 
 >> --test  
 >>> --image  
 >>> --mask  


__2.Model Hyperparameters Setting__  
Config the specification of the training process in train.py (e.g. epochs, steps) and run python train.py  
__3.Model Evaluation__    
After the training process is complete, the h5 file with saved weight will generated at the root folder, please run the evaluation.py to evaluate the performance of the model.  
__4.Obtain the Results__  
the results of cracks segmentation will generate automatically and save at the folder in /data/test/test_results/.  
# Citation
