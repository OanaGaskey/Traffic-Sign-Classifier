# Traffic-Sign-Classifier
Traffic signs recognition using Convolutional Neural Networks.

![VolvoTrafficSign](images/VolvoTrafficSign.JPG)
Photo source: [Motornature](http://www.motornature.com/)

Traffic sign recognition is crucial for a self driving car. Even though speed limit information may be available in the map and navigation system, other signs like "no passing" or "yield" are very important when it comes to making a decision while driving autonomously. 

Car manufacturers have already introduced this feature for their high end line vehicles, using the front facing camera sensor and deploying computer vision techniques as well as machine learning based classifiers. While this is largely used in sandard vehicles for driver inforamtion in the instrument cluster or for automatic cruise control, self driving cars still rely on this technique to gather more information about traffic rules. 

For this project I am going to implement a traffic sign recognition algorithm based on European signalisation. 

In order to _teach_ a car how to recognise signs and classify them correctly, I am going to start with a data set of existing images and a corresponding set of labels. Most of the work is done _offline_ and includes processing the images, designing and training a convolutional neural network. 

The trained convolutional neural network is then used _online_, meaning that it runs in real time on the car to recognize traffic signs that it sees for the first time.

