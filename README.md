This GitHub repository contains the codebase for a final project in Artificial Intelligence class. My project is about detecting black holes in videos of stars in space using AI models and ML algorithms.
Platform developed using Python and was implemented in a website using HTML code. 

The project is divided into three main parts. The first one is YOLOv8 which is an object detection algorithm that is based on CNN. This algorithm detects multiple stars in each frame of the video. The second part is the data analytics that convert the raw data from the YOLOv8 model into usable insights on the star's motion. The third part is an ANN model that determines if a star is orbiting a black hole using key factors of the star's motion. Before I can explain about the project, it is important to understand what effects black holes have on nearby stars and what differentiates black holes from any other massive object in space.

Black holes are the leftovers of stars that exploded violently. Since the mass of a black hole is concentrated into a small point, the gravitational field of black holes is powerful enough to drag stars towards them. One of the ways of detecting black holes is by spectating the motion of stars in space, since when stars are near a black hole they are forced to orbit around it. This phenomenon usually cannot happen around any other object in space since they don’t create such a strong gravitational field as black holes. By analyzing the star’s motion, we can determine if there is a black hole and calculate its location.

For the training of YOLOv8 model, I created a dataset of images if stars that were retrived from two sources. One from a simulator called "Universe Sandbox", and the other were made using an algorithm I made that take png photos and paste them into backrounds randomly. 



