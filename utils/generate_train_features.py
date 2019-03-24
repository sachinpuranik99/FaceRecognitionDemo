import numpy as np
import scipy as sp
import cv2
import os
import glob
import pickle
import h5py
import MTCNN as mtcnn


data_dump = {}
input_data_dir = "../data/"

def generate_features():
    input_files = glob.glob(input_data_dir + "/*.jpg")

    for line in input_files:
        frame = cv2.imread(line)
        _, _, features = mtcnn.process_image(frame)

        if features:
            data_dump[line[len(input_data_dir):-4]] = features[0]
        else:
            print line
            continue
    np.savez("feature_file.npz", **data_dump) 

if __name__ == "__main__":
    generate_features()
