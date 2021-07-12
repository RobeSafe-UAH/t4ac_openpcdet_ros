#!/usr/bin/python3.8

"""
Created on Mon Jul 12 10:10 2021

@author: Javier de la Peña
"""

import argparse
import glob
import sys
from functools import wraps
import numpy as np
import os

from modules.auxiliar_functions_multihead import filter_predictions

def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func
    return decorator

def main():

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Create a CSV file with the LiDAR PCL and timestamps')
    parser.add_argument('-b', '--bin_files', help='Path to the bin folder containing the PCLs',
                        default='/home/robesafe/t4ac_ws/src/t4ac_carla_simulator/ad_devkit/databases/perception/lidar/data')
    parser.add_argument('-t', '--timestamps', help='Path to the bin folder containing the PCLs',
                        default='/home/robesafe/t4ac_ws/src/t4ac_carla_simulator/ad_devkit/databases/perception/lidar/timestamp.txt')
    parser.add_argument('-o', '--openpcdet', help='Path to the OpenPCDETs library',
                        default='/home/robesafe/libraries/OpenPCDet')
    parser.add_argument('-m', '--model', help='Path to the model to use',
                        default='/home/robesafe/models/pointpillars_multihead/pp_multihead_nds5823_updated.pth')
    parser.add_argument('-cfg', '--cfg_model', help='Path to the config file of the model',
                        default='/home/robesafe/models/pointpillars_multihead/cbgs_pp_multihead.yaml')
    parser.add_argument('-csv', '--csv_lidar', help='Path of the CSV with LiDAR detections',
                        default='/home/robesafe/t4ac_ws/src/t4ac_carla_simulator/ad_devkit/databases/perception/lidar/detections.csv')
    args = parser.parse_args()

    # Add the OpenPCDETs library and modify Procesor_ROS_nuScenes
    sys.path.append(args.openpcdet)
    from modules.processor_ros_nuscenes import Processor_ROS_nuScenes

    @add_method(Processor_ROS_nuScenes)
    def new_pcl(self, bin_file, timestamp):

        # Open bin file and read the PCL with format x, y, z, intensity all float32
        with open(bin_file, 'rb') as f:
            pcl = f.read()
            pcl = np.frombuffer(pcl, dtype=np.float32)
            pcl = pcl.reshape((-1, 4))

        mask = ~(np.logical_and(abs(pcl[:,1]) < 2, abs(pcl[:,0]) < 2))
        pcl = pcl[mask]
        pcl[...,3] *= 0
        points = np.zeros((len(pcl[:,0]),5))
        points[:,0:4] = pcl
        self.actual_time = timestamp
        if(self.pcl_queue.full()):
            self.pcl_queue.get()
        self.pcl_queue.put((timestamp, points))

    # Initialize PCLs processor
    processor = Processor_ROS_nuScenes(args.cfg_model, args.model)
    processor.initilize_network()

    # Iterate over the bin files and timestamps
    time_stamps = []
    with open(args.timestamps) as f:
        time_stamps = f.readlines()

    # Remove previus CSV
    if(os.path.exists(args.csv_lidar)):
        os.remove(args.csv_lidar)

    # Create CSV
    with open(args.csv_lidar, 'a') as f:
        f.write("frame,timestamp,id,type,alpha,left,top,right,bottom,l,w,h,x,y,z,rotation_z,vx,vy,vz,score\n")

        frame = 0
        # Infinite loop to process the PCLs
        for bin_file, timestamp in zip(sorted(glob.glob(args.bin_files + '/*.bin')), time_stamps):
            timestamp = float(timestamp[:len(timestamp)-2])
            processor.new_pcl(bin_file, timestamp)
            pred_dicts = processor.inference()
            pred_boxes, pred_scores, pred_labels = filter_predictions(pred_dicts, True)

            # Get row values
            common_row = [str(frame),str(timestamp),'-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1','-1']
            len_pred_boxes = len(pred_boxes)
            row = []
            for i in range(len_pred_boxes):
                row = list(common_row)

                # Get the label
                if (pred_labels[i] != 9):
                    row[3] = str('Car')
                else:
                    row[3] = str('Pedestrian')

                # Get the bounding box
                row[9] = pred_boxes[i][3]
                row[10] = pred_boxes[i][5]
                row[11] = pred_boxes[i][4]
                row[12] = pred_boxes[i][0]
                row[13] = pred_boxes[i][1]
                row[14] = pred_boxes[i][2]
                row[15] = pred_boxes[i][6]
                row[16] = pred_boxes[i][7]
                row[17] = pred_boxes[i][8]

                # Get score
                row[19] = pred_scores[i]
            
            # Write the row
            for i in range(len(row)):
                f.write(str(row[i]))
                if(i < len(row)-1):
                    f.write(',')

            f.write("\n")
            frame += 1

if __name__ == '__main__':

    main()