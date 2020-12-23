import tensorflow as
from tensorflow.keras.preprocessing import image_dataset_from_directory


if __name__ == "__main__":
  path_1 , label_1 = "..\database\covid" , "covided"  # covid-much
  path_2 , label_2 = "..\database\\normal" , "sane"   # covid-free
  covid_DB = tf.keras.preprocessing.image_dataset_from_directory(path_1)
  sane_DB = tf.keras.preprocessing.image_dataset_from_directory(path_2)
