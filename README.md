
# **Project: CNN and OpenCV in Real Time**
# **Expel Pigeons with AI Drone** 

In this project we implements **Movements Detection**, **Object Detection** and **Object Tracking** algorithm. 
We do it by recognize movement in the frame, detect the object and tracking after it in real time from camera. 
We use **Resnet50** model with **Transfer Learning** technique on **CIFAR10** Dataset that we trained and performs **95% accuracy.**

The first part of the project is about **Deep Learning** and focus in **Convolutional Neural Network** and related subjects like **Overfitting, 
Transfer Learning** including **Features Extractor** and **Fine Tuning** and **Import Dataset From Binary Files.**
The second part of the project is about **Computer Vision** using libraries like **OpenCV, Numpy, h5py** etc'.

In the begining of the project we test different type of models, with different types of techniques, to find the model that fit most to our goal, recognize birds.
We found that the model with the best result is **ResNet50** with **Transfer Learning** using **Fine Tuning**, wich bring us result of **95% accuracy** on **CIFAR-10** dataset.

After that we build the second part of the object, the front. The app running between two states: **Detection** and **Tracking.** 
With this algorithm, instead of using Resnet50 prediction for every frame, we detect the object only once in the beginning and then tracking after it. 
For example, if we are reading 2000 frames it can be the different between running the model 2000 times or only once. 
We are also starting to detect object only when we recognize a movement in the frame, it help the app not running in cases of quiet frames.

First, we recognize a movement in the frame, then we cut the suspect part in the frame and send it to the model for prediction. 
We detect the object by using our model prediction and then we start tracking after it. 
when the tracking is over, we start to detect objects again. We store all the detections events in format of date and time and show it on the screen.
We also save the frames with detection event in folder and a text file that contain all the detections events.

The system also suggest a drone mode which is send the drone automatic to patrol every time we detect a bird to expel it.
The drone system is a SDK we built and allow us to control the drone from far by remote control with the keyboard. 
It also display a board (radar) that synchronize with the drone location and dinstace when the drone flight.
The SDK also gives us information of the drone like speed, distance, camera, capture image, build-in tracks, detection and tracking etc.

The application display one window divided to six screens:
  1.	Main Frame that we read from camera.
  2.	Information Frame that includes detection events and application information.
  3.	Tracking Frame with the object's coordinates, score, prediction time and FPS.
  4.	Drone camera Frame that we read from camera with drone speed, angle, battery etc.
  5.	Scores Frame that shows us the scores for every label.
  6.	Drone radar which is synchronize with drone movements and it distance.
  
The application using **TensorFlow** with **Keras** by **Python**.


.
.
.

**Will update soon**

Let's see a short video of our application:





Example of folder with the saved frames from detection event:
 
 

Example of the text file with detections events and application information:
