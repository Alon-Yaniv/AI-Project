"""
Routes and views for the flask application.
"""

from datetime import datetime
from pyexpat import model
from typing import final
from flask import render_template, request
from tensorflow.python.saved_model.save import metrics
from Final_Project import app
from flask import Flask, url_for
import tensorflow as tf
from ultralytics import YOLO
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from PIL import Image
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from colorama import Fore, Back, Style
import cv2



#The function recives two point A, B and returns a vector AB->
def create_vector(point1, point2):
  vector = [point2[0] - point1[0], point2[1] - point1[1]]
  return vector
#The function recives two diffrent vectors, and returns the angle between them minus 180 degrees
def get_angle_between_two_vectors(vector1, vector2):
  skalar = float(vector1[0]*vector2[0] + vector1[1]*vector2[1])
  length_of_vector1 = float(math.sqrt(pow(vector1[0], 2) + pow(vector1[1], 2)))
  length_of_vector2 = float(math.sqrt(pow(vector2[0], 2) + pow(vector2[1], 2)))
  angle = round(((math.acos(round((skalar/(length_of_vector1 * length_of_vector2)),4)) / math.pi) * 180),4)
  #print("angle", angle)
  return angle


# @title Image Class
#A class for image with information about it
class Img():

  def __init__(self, img):
    self.img = img
    self.img_np_array = np.array(self.img, dtype = np.uint8)

  def get_img(self):
    return self.img

  def get_img_np_array(self):
    return self.img_np_array

  def set_img(self, img):
    self.img = img
    self.img_np_array = np.array(self.img, dtype = np.uint8)

  def rescale(self, scale = 0.2):
    width, height = self.img.size
    self.img = self.img.resize((int(width * scale), int(height * scale)))
    self.img_np_array = np.array(self.img, dtype = np.uint8)
    return self.img

  def fix_colors(self):
    self.img_np_array = cv2.cvtColor(self.img_np_array, cv2.COLOR_BGR2RGB)
    self.img = Image.fromarray(self.img_np_array)
    return self.img

  def save_img(self, path):
    self.img.save(path)

  """def show_img(self):
    display(self.img)"""

# @title Video Class
#A class for video with infromation about the it
class Video():

  def __init__(self, video_path):
    self.video_path = video_path
    self.video = cv2.VideoCapture(video_path)
    self.video_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

  def get_video_path(self):
    return self.video_path

  def get_video_length(self):
    return self.video_length

  def get_video(self):
    return self.video

  def set_video_path(self, video_path):
    self.video_path = video_path
    self.video = cv2.VideoCapture(video_path)
    self.video_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

  def get_frames(self):
    arr = []
    for i in range(self.video_length):
      frame = Img(Image.fromarray(self.video.read()[1]))
      frame.rescale(1)
      arr.append(frame.fix_colors())

    return arr

  def save_frames(self, folder_path, amount = 0):
    if(amount == 0):
      amount = self.video_length

    arr = []
    for i in range(amount):
      frame = Img(Image.fromarray(self.video.read()[1]))
      frame.rescale(0.08)
      cv2.imwrite(f"{folder_path}/frame{i}.jpg", frame.get_img_np_array())
      arr.append(frame.fix_colors())

    return arr

# @title Star
class Star():

  def __init__(self):
    self.locations = np.array([])

  def add_frame_location(self, location):
    if(len(self.locations) == len(np.array([]))):
      self.locations = np.array([location])
    else:
      self.locations = np.append(self.locations, [location], axis = 0)
      
  def get_first_frame_location(self):
    return self.first_frame_location

  def get_locations(self):
    return self.locations

  def get_location_by_index(self, index):
    return self.locations[index]

  #The function return True if the star's orbit is a full one at least once or False otherwise
  def is_full_orbit(self):
    #It does it by calculating the delta distances between the first star and the rest (each) and if we see that the distance is closing on zero again (on the graph) we can know that the star has at least one full orbit. In a nutshell, we need to find the minimum point of the function, in this point the sign of two distances changes from positive to negative
    first_location = self.locations[0]
    distances = []
    flag_is_up = True
    distance1 = math.sqrt(pow((first_location[0] - self.locations[10][0]), 2) + pow((first_location[1] - self.locations[10][1]), 2))
    self.x = int(len(self.locations) /  (3 *(distance1) * len(str(len(self.locations)))))
    if(self.x == 0):
      self.x = 1
    for loc_index in range (0 , len(self.locations), self.x):
      current_location = self.locations[loc_index]
      distance = math.sqrt(pow((first_location[0] - current_location[0]), 2) + pow((first_location[1] - current_location[1]), 2))
      distances.append(distance)
    self.distances = distances
    min_distance = 99999999.0
    final_step = 0
    for step in range(2, 10):
      flag_is_full_orbit = False
      flag_is_up = True
      flag_need_validation = False
      for i in range(step, len(distances)):
        delta = distances[i]- distances[i-step]
        if(flag_need_validation == True):
          if(delta > 0):
            flag_is_up = True
          else:
            flag_is_full_orbit = True
          flag_need_validation = False
        if(delta > 0 and flag_is_full_orbit == True):
          if(min_distance > distances[i-step]):
            min_distance = distances[i-step]
            final_index_of_full_orbit = i - step
            final_step = step
        if(min_distance < 12):
          return True
        if(delta < 0):
          flag_is_up = False
          flag_need_validation = True
    return False

  def show_distances(self):
    plt.plot(range(0, len(self.locations),self.x), self.distances, "ro")
    plt.xlabel("Frames")
    plt.ylabel("Distance")
    plt.show()

  def calc_distance(self, index1, index2):
    loc1 = self.locations[index1]
    loc2 = self.locations[index2]
    distance = math.sqrt(pow((loc1[0] - loc2[0]), 2) + pow((loc1[1] - loc2[1])))
    return distance

  def show_orbit(self, img, img_size):
    R = random.randint(0, 256)
    G = random.randint(0, 256)
    B = random.randint(0, 256)
    color = (R, G, B)
    for frame in range(len(self.locations)):
      loc = self.locations[frame]
      cv2.circle(img , (int(loc[0]), int(loc[1])),radius= 2, color = color ,thickness= 2)
    return img
# @title Stars
class Stars():

  def __init__(self, stars_array):
    self.stars_array = stars_array

  def get_stars_array(self):
    return self.stars_array

  def show_stars(self, img, img_size):
    colors = []
    for color in range(len(self.stars_array)):
      R = random.randint(0, 256)
      G = random.randint(0, 256)
      B = random.randint(0, 256)
      colors.append((R, G, B))

    for i in range(len(self.stars_array)):
      for star in range(len(self.stars_array[i].get_locations())):
        loc = self.stars_array[i].get_locations()[star]
        cv2.circle(img , (int(loc[0]), int(loc[1])),radius= 2, color = colors[i],thickness= 2)

    #cv2.imshow(img)
    return img
# @title Orbit
class Orbit():
#Transform the locations array to one full orbit of locations (if it's possible)
  def __init__(self, locations):
    #We need to find the smallest distance from point 0
    final_index_of_full_orbit = 0
    loc1 = locations[0]
    distances= []
    x_distances = []
    for i in range(1, len(locations),1):
      loc2 = locations[i]
      distance = math.sqrt(pow((loc1[0] - loc2[0]), 2) + pow((loc1[1] - loc2[1]), 2))
      distances.append(distance)
      x_distances.append(i)

    final_delta_distances = []
    final_x_delta_distances = []
    min_distance = 99999999.0
    final_step = 0
    validation_distance = distances[10]
    #Step is determintated
    for step in range(2, 15):
      flag_is_up = True
      delta_distances = []
      x_delta_distances = []
      for i in range(step+5, len(distances),step):
        delta = distances[i]- distances[i-step]
        delta_distances.append(delta)
        x_delta_distances.append(i)
        if(delta > 0 and flag_is_up == False and distances[i-step] < validation_distance):
          if(min_distance > distances[i-step]):
            min_distance = distances[i-step]
            final_index_of_full_orbit = i - step
            final_delta_distances = delta_distances
            final_x_delta_distances = x_delta_distances
            final_step = step
          break
        if(delta < 0):
          flag_is_up = False
    print("Final step: ", final_step)
    self.final_index_of_full_orbit = final_index_of_full_orbit
    self.locations = locations[0:final_index_of_full_orbit]
    self.distances = distances
    self.delta_distances = final_delta_distances

  def set_locations(self, locations):
    self.locations = locations

  def get_locations(self):
    return self.locations

  def show_distances(self):
    plt.plot(range(len(self.distances)), self.distances, "ro")
    plt.plot(self.final_index_of_full_orbit, self.distances[self.final_index_of_full_orbit], 'bo')
    plt.xlabel("Frames")
    plt.ylabel("Distance")
    plt.show()

  def show_delta_distances(self):
    plt.plot(range(len(self.delta_distances)), self.delta_distances, "ro")
    plt.xlabel("Frames")
    plt.ylabel("Delta Distance")
    plt.show()

  def show_orbit(self, img, img_size):
    R = random.randint(0, 256)
    G = random.randint(0, 256)
    B = random.randint(0, 256)
    color = (R, G, B)
    for frame in range(len(self.locations)):
      loc = self.locations[frame]
      cv2.circle(img , (int(loc[0]), int(loc[1])),radius= 2, color = color ,thickness= 2)
    #cv2.imshow(img)
    return img

  def get_a_of_an_ellipse(self):
    max_distance = 0.0
    index_star0_max = 0
    index_star1_max_final = 0
    for star0 in range(len(self.locations)):
      max_dis_in_epoch = 0
      index_star1_max = 0
      for star1 in range(len(self.locations)):
        if(star1 != star0):
          distance = math.sqrt(pow((self.locations[star0][0] - self.locations[star1][0]), 2) + pow((self.locations[star0][1] - self.locations[star1][1]),2))
          if(distance > max_dis_in_epoch):
            max_dis_in_epoch = distance
            index_star1_max = star1
      if(max_distance < max_dis_in_epoch):
        max_distance = max_dis_in_epoch
        index_star1_max_final = index_star1_max
        index_star0_max = star0

    self.a_point1_index = index_star0_max
    self.a_point2_index = index_star1_max_final
    return self.a_point1_index, self.a_point2_index

  def get_b_of_an_ellipse(self):
    point_zero = [(self.locations[self.a_point1_index][0] + self.locations[self.a_point2_index][0])/2, (self.locations[self.a_point1_index][1] + self.locations[self.a_point2_index][1])/2]
    min_number = 999.0
    first_vector = create_vector(point_zero, self.locations[self.a_point1_index])
    for i in range(1, len(self.locations)):
      current_vector = create_vector(point_zero, self.locations[i])
      angle = get_angle_between_two_vectors(first_vector, current_vector)
      if(abs(angle - 90) < min_number):
        min_number = abs(angle-90)
        max_angle_index = i

    self.b_point1_index = max_angle_index
    return self.b_point1_index

  def get_the_locations_of_the_black_hole(self):
    point_zero = [(self.locations[self.a_point1_index][0] + self.locations[self.a_point2_index][0])/2, (self.locations[self.a_point1_index][1] + self.locations[self.a_point2_index][1])/2]
    point_a1 = self.locations[self.a_point1_index]
    point_a2 = self.locations[self.a_point2_index]
    point_b1 = self.locations[self.b_point1_index]
    a = math.sqrt(pow((point_a1[0] - point_zero[0]), 2) + pow((point_a1[1] - point_zero[1]), 2))
    b = math.sqrt(pow((point_b1[0] - point_zero[0]), 2) + pow((point_b1[1] - point_zero[1]), 2))
    c = math.sqrt(pow(a, 2) - pow(b, 2))
    l = c
    k = abs(a-c)
    x1 = ((point_a1[0] * l) + (point_zero[0] * k)) / (l + k)
    y1 = ((point_a1[1] * l) + (point_zero[1] * k)) / (l + k)
    x2 = ((point_a2[0] * l) + (point_zero[0] * k)) / (l + k)
    y2 = ((point_a2[1] * l) + (point_zero[1] * k)) / (l + k)
    point1 = [x1, y1]
    point2 = [x2, y2]

    #We need to decide which point represent our black hole, and it's the one that closer to the heighest speed location
    #Find the heighest speed locations:
    black_hole_location = []
    max_distance = 0.0
    max_distance_index = 0
    last_location = self.locations[0]
    for i in range(1, len(self.locations)):
      current_location = self.locations[i]
      distance = math.sqrt(pow((last_location[0] - current_location[0]), 2) + pow((last_location[1] - current_location[1]), 2))
      if(distance > max_distance):
        max_distance = distance
        max_distance_index = i
      last_location = current_location
    #Which point is closer:
    heighest_speed_location = self.locations[max_distance_index]
    distance_from_point1 = math.sqrt(pow((point1[0] - heighest_speed_location[0]), 2) + pow((point1[1] - heighest_speed_location[1]), 2))
    distance_from_point2 = math.sqrt(pow((point2[0] - heighest_speed_location[0]), 2) + pow((point2[1] - heighest_speed_location[1]), 2))
    if(distance_from_point1 < distance_from_point2):
      black_hole_location = point1
    else:
      black_hole_location = point2

    self.black_hole_location = black_hole_location
    return black_hole_location


  def show_orbit_with_black_hole(self, img_size):
    point_zero = [(self.locations[self.a_point1_index][0] + self.locations[self.a_point2_index][0])/2, (self.locations[self.a_point1_index][1] + self.locations[self.a_point2_index][1])/2]
    img = np.zeros(img_size, dtype = np.uint8)
    R = random.randint(0, 256)
    G = random.randint(0, 256)
    B = random.randint(0, 256)
    color = (R, G, B)
    for frame in range(len(self.locations)):
      loc = self.locations[frame]
      cv2.circle(img , (int(loc[0]), int(loc[1])),radius= 2, color = color ,thickness= 2)
    cv2.circle(img, (int(self.locations[self.a_point1_index][0]), int(self.locations[self.a_point1_index][1])), radius = 2,color = (255,255,255), thickness = 2)
    cv2.circle(img, (int(self.locations[self.a_point2_index][0]), int(self.locations[self.a_point2_index][1])), radius = 2,color = (255,255,255), thickness = 2)
    cv2.circle(img, (int(point_zero[0]), int(point_zero[1])), radius = 2,color = (255,255,0), thickness = 2)
    cv2.circle(img, (int(self.locations[self.b_point1_index][0]), int(self.locations[self.b_point1_index][1])), radius = 2,color = (0,255,0), thickness = 2)
    #cv2.circle(img, (int(self.locations[min_index_1][0]), int(self.locations[min_index_1][1])), radius = 2,color = (255,255,0), thickness = 2)
    cv2.circle(img, (int(self.black_hole_location[0]), int(self.black_hole_location[1])), radius = 2,color = (0,0,255), thickness = 2)
    #cv2.imshow(img)

# @title Orbits
class Orbits():
  def __init__(self, orbits_array):
    self.orbits_array = orbits_array

  def add_orbit(self, orbit):
    self.orbits_array.append(orbit)

  def show_orbits_and_black_hole(self, img, img_size):
    #Get black holes array
    #For every orbit we need to find it's black hole
    if(len(self.orbits_array) == 0):
      return None
    black_hole_array = []
    for orbit_index in range(len(self.orbits_array)):
      current_orbit = self.orbits_array[orbit_index]
      current_orbit.get_a_of_an_ellipse()
      current_orbit.get_b_of_an_ellipse()
      black_hole_array.append(current_orbit.get_the_locations_of_the_black_hole())
    #Find the average black hole
    average_black_hole_location = []
    x = 0
    y = 0
    for black_hole in black_hole_array:
      x += black_hole[0]
      y += black_hole[1]
    average_black_hole_location.append(x/len(black_hole_array))
    average_black_hole_location.append(y/len(black_hole_array))

  #Show the orbits and black hole
    colors = []
    for color_index in range(len(self.orbits_array)):
      R = random.randint(0, 256)
      G = random.randint(0, 256)
      B = random.randint(0, 256)
      colors.append((R, G, B))
    for orbit_index in range(len(self.orbits_array)):
      current_orbit = self.orbits_array[orbit_index]
      current_orbit_locations = current_orbit.get_locations()
      for frame in range(len(current_orbit_locations)):
        loc = current_orbit_locations[frame]
        cv2.circle(img , (int(loc[0]), int(loc[1])),radius= 2, color = colors[orbit_index] ,thickness= 2)

    cv2.circle(img, (int(average_black_hole_location[0]), int(average_black_hole_location[1])), radius = 2,color = (0,0,255), thickness = 2)
    return img

# @title Yolo Class
class yolo_class:

#BUild a custom model
  def __init__(self, model_path = None):
    if(model_path == None):
      self.model_path = None
      self.model = None
      print("Need training")
    else:
      self.model_path = model_path
      self.model = YOLO(self.model_path, task = "detect")

  def get_model(self):
    return self.model

  def get_model_path(self):
    return self.model_path

  #Train a yolo model
  def train(self, data_path, model_type , amount_of_epochs, image_size, where_to_save):
    self.model_path = model_type
    self.model = YOLO(self.model_path)
    self.model.train(data = data_path, epochs = amount_of_epochs, imgsz = image_size, project = where_to_save)

  #Predict an image
  def predict_img(self, image_path):
    result = self.model.predict(cv2.imread(image_path))
    return result

  #Predict a video
  def predict_vid(self, video_path, save = True):
    model = YOLO(self.model_path)
    if(save):
      results = model(video_path, save = True)
    else:
      results = model(video_path)
    return results

  #Predict a file
  def predict_img_files(self, file_path):
    results = self.model(file_path, stream = True)
    return results

  def show_prediction(self, results, image_path):
    img = cv2.imread(image_path)
    for result in results:
      boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
      for box in boxes:  # Iterate over boxes
        r = box.xyxy[0].astype(int)  # Get corner points as int
        class_id = int(box.cls[0])  # Get class ID
        class_name = self.model.names[class_id]  # Get class name using the class ID
        #print(f"Class: {class_name}, Box: {r}")  # Print class name and box coordinates
        cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 2)  # Draw boxes on the image
        #center = get_central_mass(r.copy())
        #cv2.circle(img, (int(center[0]), int(center[1])),radius= 5, color =(0,255,0),thickness= 2)
    cv2.imshow(img)

# @title Excel File
class Excel_File():

  def __init__(self, file_path):
    self.file_path = file_path
    self.dict_data = {}
    if(os.path.exists(self.file_path) == False):
      d = pd.DataFrame(self.dict_data)
      d.to_excel(self.file_path)
    else:
      data = pd.read_excel(self.file_path)
      self.dict_data = data.to_dict()
      del self.dict_data["Unnamed: 0"]

  #The inputs should be type  array
  #The data should be a two dimantion array, that each row represent diffrent types of data (keys)
  def create_or_add_data(self, data_names, data):
  #The data needs to be type array, and the data should be a two dimantions array, which each row represent different types of data (keys) ***Without new keys and their data. The function return the amount of added data.
    length_of_added_data = 0
    if(len(self.dict_data) == 0):
      for key_index in range(len(data_names)):
        dict_new = {data_names[key_index] :{}}
        arr = []
        for data_index in range(len(data[0])):
          arr.append([data_index , data[key_index][data_index]])
        dict_new[data_names[key_index]].update(arr)
        self.dict_data = self.dict_data | dict_new
    else:
      data_arr = []
      length_of_existing_data = len(self.dict_data[data_names[0]])
      for key_index in range(len(data_names)):
        arr = []
        for data_index in range(len(data[0])):
          arr.append([data_index + length_of_existing_data, data[key_index][data_index]])
        data_arr.append(arr)
    #Check if there is identical data
      for data_index in range(len(data_arr[0])):
        one_data = [] # Represent a single sample from the data
        for key_index in range(len(data_arr)):
          if(data_arr[key_index][data_index][1] == None):
            break
          else:
            one_data.append(data_arr[key_index][data_index][1])

        #Convert old data to data array
        old_data = pd.read_excel(self.file_path).to_numpy()
        for dict_index in range(len(old_data)):
          old_one_data = [] # Represent single sample from the old data
          for key_index in range(len(data_names)):
            old_one_data.append(old_data[dict_index][key_index+1])
          if(old_one_data == one_data): # Checking if ther'e equale
            for key_index in range(len(data_names)):
              data_arr[key_index][data_index] = None
      #Compare to old data
      for key_index in range(len(data_names)):
        arr = []
        for arr_index in range(len(data_arr[key_index])):
          if(data_arr[key_index][arr_index] != None):
            arr.append(data_arr[key_index][arr_index])
        length_of_added_data = len(arr)
        self.dict_data[data_names[key_index]].update(arr) # Adding the different
    return length_of_added_data

  def get_data_as_dict(self):
    return self.dict_data

  def Create_or_update_excel_file(self):
    d = pd.DataFrame(self.dict_data)
    d.to_excel(self.file_path)

  #The data is in type numpy array
  def get_data_as_np_array(self):
    data = pd.read_excel(self.file_path)
    return data.to_numpy()


  #from_index_to_index is array with two values - index of start, and index of end
  def delete_certain_data(self,data_names, from_index_to_index):
    old_data = pd.read_excel(self.file_path)
    del old_data["Unnamed: 0"]
    print(old_data._is_copy)
    if(len(self.dict_data) != 0):
      old_data.drop(from_index_to_index, inplace = True)
      old_data = old_data.reset_index(drop = True)#drop = True ensure that another column won't be added
      print(old_data)
      old_data.to_excel(self.file_path)
    else:
      print("There is nothing to delete")




@app.route('/')
@app.route('/Main Page')
def main_page():
    """Renders the home page."""
    return render_template(
        'main_page.html',
        title='Main Page',
        year=datetime.now().year,
        message = "Welcome to Alon's space website!"
    )

def get_boxes_to_stars_for_video(video_results, real_amount_of_stars_in_a_video):
  stars_locations = []
  count_frames = 0
  for frame in video_results:
    boxes = frame.boxes.cpu().numpy()
    current_locations = []
    need_to_add_to_stars_locations = [] #An array that should consist the three location that were classified to each star
    for box in boxes:
      location = box.xyxy[0].astype(float)
      current_locations.append(get_central_mass(location))
    #Set the central mass location of each star
    #We need to differentiate between three scenarios. First when it's the first frame, second and the rest of the frames. ***The first and second must be predicted perfectly and each prediction must be far away from eachother for best results (at least 30 pixels)
    #In the first frame we will insert the stars_locations array the locations.
    #In the second frame we can determine the next star location by what is the closest location to it
    #In the the third frame and the rest we can know the vector of the last two frames, and determinate the most accurate location by the angle, closeness to each other and by speed.
    if(count_frames == 0): #We are on the first frame.
        if(real_amount_of_stars_in_a_video != int(len(current_locations))):
            print("Prediction isn't accurate, need more accurate prediction - first frame")
            return None
        else:
            need_to_add_to_stars_locations = current_locations.copy()
    elif(count_frames == 1): #We are on the second frame.
        if(real_amount_of_stars_in_a_video != int(len(current_locations))):
            print("Prediction isn't accurate, need more accurate prediction - second frame")
            return None
        else:
            for location_index in range(len(current_locations)):
                closest_index, d_min = who_is_closest_location(current_locations.copy(), stars_locations[-1][location_index])
                need_to_add_to_stars_locations.append(current_locations[closest_index])
    else: #We are in the rest of the frames.
        if(real_amount_of_stars_in_a_video == len(current_locations)):
            for location_index in range((len(current_locations))):
                closest_index, d_min = who_is_closest_location(current_locations.copy(), stars_locations[-1][location_index])
                need_to_add_to_stars_locations.append(current_locations[closest_index])
        elif(real_amount_of_stars_in_a_video < len(current_locations)):
            for location_index in range(real_amount_of_stars_in_a_video):
                closest_index, d_min = who_is_closest_location(current_locations.copy(), stars_locations[-1][location_index])
                need_to_add_to_stars_locations.append(current_locations[closest_index])
        elif(real_amount_of_stars_in_a_video > len(current_locations)):
            for location_index in range(real_amount_of_stars_in_a_video):
                closest_index, d_min = who_is_closest_location(current_locations.copy(), stars_locations[-1][location_index])
                if(d_min > 4):
                    need_to_add_to_stars_locations.append(stars_locations[-1][location_index])
                else:
                    need_to_add_to_stars_locations.append(current_locations[closest_index])

    stars_locations.append(need_to_add_to_stars_locations)
    count_frames += 1
  return stars_locations

#The function recives a loactions array and a star index that we want to check, and returns the index of the closest one
def who_is_closest_location(locations_array, previous_star_location):
  closest_index = 0
  min_distance = 9999999999
  for i in range(len(locations_array)):
    star_point_x = previous_star_location[0]
    star_point_y = previous_star_location[1]
    point_x = locations_array[i][0]
    point_y = locations_array[i][1]
    distance = math.sqrt(pow((star_point_x - point_x), 2) + pow((star_point_y - point_y), 2))
    #print("distance", distance)
    if(min_distance > distance):
      min_distance = distance
      closest_index = i
  return closest_index, min_distance


#The function recives loactions ((x,y), (x,y)) and return the central mass location of the star (x,y)
def get_central_mass(location):
  point1x = location[0]
  point1y = location[1]
  point2x = location[2]
  point2y = location[3]
  central_pointx = float((point1x + point2x)/2)
  central_pointy =  float((point1y + point2y)/2)
  central_point = [central_pointx, central_pointy]
  return central_point

#The function detect if there are copies of location and return True of none were found or False otherwise
def detect_mistakes_in_stars_locations(stars_array):
  for amount_of_frames in range(len(stars_array[0].get_locations())):
    if(len(stars_array) > 1):
      locations = []
      for i in range(len(stars_array)):
        locations.append(stars_array[i].get_locations()[amount_of_frames])
      for loc1 in range(len(locations)):
        for loc_rest in range(loc1+1, len(locations)):
          if((locations[loc1][0] == locations[loc_rest][0]) and (locations[loc1][1] == locations[loc_rest][1])):
            return False
  return True


#The function transform the location array to organized Stars array (from type list to type Star)
def get_location_to_Stars_array(locations):
  stars_locations = []
  for amount in range(len(locations[0])):
    stars_locations.append(Star())
  for location in locations:
    for star in range(len(location)):
      stars_locations[star].add_frame_location(location[star])

  return stars_locations

#The function recives type *stars* array, and claculate the alpha between two vectors on a non full orbit star, and returns the index of the curve and the slpha varient
def cruve_orbit_alpha_varient(star):
  star_locations = star.get_locations()
  first_location = star_locations[0]
  middle_location = star_locations[len(star_locations)//2]
  last_location = star_locations[-1]
  first_vector = create_vector(middle_location, first_location)
  second_vector = create_vector(middle_location, last_location)
  alpha = get_angle_between_two_vectors(first_vector, second_vector)
  return alpha


def total_distance_in_star_orbit(star):
  star_locations = star.get_locations()
  total_distance = 0
  step = 5
  for loc_index in range(step, len(star_locations), step):
    distance = math.sqrt(pow((star_locations[loc_index][0] - star_locations[loc_index - step][0]), 2) + pow((star_locations[loc_index][1] - star_locations[loc_index - step][1]), 2))
    total_distance += distance
  total_distance = round(total_distance,3)
  return  total_distance

def train_model(epochs = 500, learning_rate = 0.01):
    excel = Excel_File("C:/Users/alony/OneDrive/Documents/OneDrive/Alon website/Final project in AI/Final_Project/AI Data.xlsx")
    data = excel.get_data_as_np_array()
    #key 0 is index, and we don't need it.
    features = np.delete(data, 0, axis = 1)
    features = np.delete(features, 3, axis = 1)

    #Key index 4 is our outputs and we need them separately
    outputs = data[:, 4]
    #print(features)
    model = tf.keras.Sequential([tf.keras.layers.Dense(60,activation='relu',input_shape=(3,)), tf.keras.layers.Dense(120,activation='relu'), tf.keras.layers.Dense(1,activation='sigmoid')])
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), loss ="binary_crossentropy", metrics = ['accuracy'])# loss = binary_crossentropy
    history = model.fit(features,outputs,epochs=epochs)
    results = model.predict(features)
    count_correct = 0
    for i in range(len(results)):
        res = 0
        if(results[i] > 0.5):
            res = 1
        if(int(res) == int(outputs[i])):
            count_correct += 1
    model.save("model.h5")
    print("Acurracy is: ", (count_correct / len(outputs))*100)
    loss = history.history['loss']
    plt.plot(range(len(loss)), loss, "r")
    plt.show()
    plt.plot(range(len(results)), results, 'ro')
    check = []
    for i in range(len(results)):
        check.append(0.5)
    plt.plot(range(len(results)), check, 'b-')



@app.route('/upload', methods = ['Get', 'POST'])
def upload():
    if(request.method == "GET"): 
        return render_template("upload.html")
    else:
        video = request.files["video"]
        print(video)
        if(request.form.get("amount of stars") == "" or video.filename == ""):
            return render_template(
                "upload.html",
                message = "The amount of stars is not right or you didn't insert a video. Try again.",

            )
        video.save("C:/Users/alony/OneDrive/Documents/OneDrive/Alon website/Final project in AI/Final_Project/Final_Project/Final_Project/static/fonts/" + video.filename)
        model_path = "C:/Users/alony/OneDrive/Documents/OneDrive/Alon website/Final project in AI/Final_Project/yolo v8 model - v18/train/train/weights/best.pt" #best one
        yolo_model = yolo_class(model_path)
        video = Video("C:/Users/alony/OneDrive/Documents/OneDrive/Alon website/Final project in AI/Final_Project/Final_Project/Final_Project/static/fonts/" + request.files['video'].filename)
        video_prediction = yolo_model.predict_vid(video.get_video_path())

        real_amount_of_stars = int(request.form.get("amount of stars"))
        accuracy = 0
        wrong_detections = 0
        amount_of_predicted_stars_in_video = 0
        wrong_or_miss_detection = 0
        for result in video_prediction:
            boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
            wrong_or_miss_detection += abs(len(boxes) - real_amount_of_stars) #number of wrong detections in a single frame
            for box in boxes:  # Iterate over boxes
                amount_of_predicted_stars_in_video += 1
        accuracy = (100 - ((wrong_or_miss_detection/(real_amount_of_stars*video.get_video_length()))*100))
        wrong_detections = wrong_or_miss_detection
        #For each video wer'e transforming the video prediction to a type *Stars*
        video_of_stars = []
        message = ""
        video_width = int(video.get_video().get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get_video().get(cv2.CAP_PROP_FRAME_HEIGHT))
        img_size = (video_height, video_width, 3)
        stars_results = get_boxes_to_stars_for_video(video_prediction, real_amount_of_stars)
        if(stars_results == None):
            return render_template(
                "upload.html",
                message = "The amount of stars is not right! Try again.",

            )
        stars_array = get_location_to_Stars_array(stars_results)
        stars = Stars(stars_array.copy())
        #stars.show_stars(img_size)
        validity = detect_mistakes_in_stars_locations(stars_array)
        print("The Video is valid and no stars colding: ", validity)
        print("=====================================================================")
        video_of_stars = stars

        #================================================================================================================================#
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        model = tf.keras.models.load_model("C:/Users/alony/OneDrive/Documents/OneDrive/Alon website/Final project in AI/Final_Project/Final_Project/model.h5")
        print(model.summary())
        count_none_full_orbit = 0
        count_orbiting_around_a_black_hole = 0
        count_full_orbit = 0
        if(validity == True):
          orbits_array = []
          vid = video_prediction
          final_img_array =  np.zeros(img_size, dtype = np.uint8)
          is_orbiting_black_hole_final = False
          for star_index in range(len(video_of_stars.get_stars_array())):
            print("Star number ", star_index + 1, ":")
            star = video_of_stars.get_stars_array()[star_index]
            is_full_orbit = star.is_full_orbit()
            if(is_full_orbit == True):
              orbit = Orbit(star.get_locations())
              orbits_array.append(orbit)
              orbit.show_distances()
              final_img_array = orbit.show_orbit(final_img_array, img_size)
              #orbit.show_delta_distances()
              orbit.get_a_of_an_ellipse()
              orbit.get_b_of_an_ellipse()
              orbit.get_the_locations_of_the_black_hole()
              #orbit.show_orbit_with_black_hole(img_size)
              is_orbiting_black_hole_final = True
              count_full_orbit+=1
              #orbit1.show_orbit((video_height, video_width, 3))
              print("Orbit's length: ", len(orbit.get_locations()))
              print("Total frames: ", len(video_of_stars.get_stars_array()[star_index].get_locations()))
            else:
              final_img_array = star.show_orbit(final_img_array, img_size)
              count_none_full_orbit += 1
              #star.show_distances()
              #star.show_orbit(img_size)
              Angle = cruve_orbit_alpha_varient(star)
              Distance = total_distance_in_star_orbit(star)
              Delta = float(len(star.get_locations()))
              print("The star didn't made a full orbit")
              is_orbiting_black_hole = model.predict(np.array([[Angle, Distance, Delta]]))
              if(is_orbiting_black_hole >= 0.5):
                count_orbiting_around_a_black_hole += 1
                print("This star is orbiting a black hole!")
              else:
                print("This star isn't orbiting a black hole")
            print("============================================================================================================================")
            
          orbits = Orbits(orbits_array)
          if(float(count_orbiting_around_a_black_hole) >= count_none_full_orbit * 0.5):
            is_orbiting_black_hole_final = True
          if(count_full_orbit > 0):
            img = Img(Image.fromarray(orbits.show_orbits_and_black_hole(final_img_array, img_size)))
            img.fix_colors()
            img.get_img().save("C:/Users/alony/OneDrive/Documents/OneDrive/Alon website/Final project in AI/Final_Project/Final_Project/Final_Project/static/fonts/current_img.jpg")
            return render_template(
                "predict.html",
                message = "This star system is orbiting a black hole!",
                image_url = url_for('static', filename='fonts/current_img.jpg'),
                details_full_orbit = "According to this video, there are " + str(count_full_orbit) + " amount of stars that made a full orbit around the black hole.",
                details_non_full_orbit = "According to this video, there are " + str(count_none_full_orbit) + " amount of stars that didn't make a full orbit around the black hole.",
                accuracy_of_video = "The accuracy of the prediction of the video is: " + str(accuracy) + "%.",
            )
          elif(is_orbiting_black_hole_final == True):
            img = Img(Image.fromarray(final_img_array))
            img.get_img().save("C:/Users/alony/OneDrive/Documents/OneDrive/Alon website/Final project in AI/Final_Project/Final_Project/Final_Project/static/fonts/current_img.jpg")
            return render_template(
                "predict.html",
                message = "This star system is orbiting a black hole! But the system can't calculate it's position.",
                image_url = url_for('static', filename='fonts/current_img.jpg'),
                details_full_orbit = "According to this video, there are " + str(count_full_orbit) + " amount of stars that made a full orbit around the black hole.",
                details_non_full_orbit = "According to this video, there are " + str(count_none_full_orbit) + " amount of stars that didn't make a full orbit around the black hole.",
                accuracy_of_video = "The accuracy of the prediction of the video is: " + str(accuracy) + "%.",
            )
          else:
            img = Img(Image.fromarray(final_img_array))
            img.get_img().save("C:/Users/alony/OneDrive/Documents/OneDrive/Alon website/Final project in AI/Final_Project/Final_Project/Final_Project/static/fonts/current_img.jpg")
            return render_template(
                "predict.html",
                message = "This star system isn't orbiting a black hole :-(",
                image_url = url_for('static', filename='fonts/current_img.jpg'),
                details_non_full_orbit = "According to this video, there are " + str(count_none_full_orbit) + " amount of stars in the video that aren't orbiting a black hole.",
                accuracy_of_video = "The accuracy of the prediction of the video is: " + str(accuracy) + "%.",
            )
        else:
            return render_template(
                "predict.html",
                message = "The validity of the video is bad, it seems like stars have colided",
                    
            )
        #================================================================================================================================#

        