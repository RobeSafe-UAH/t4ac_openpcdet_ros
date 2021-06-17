# General use imports
from pathlib import Path
import time
from queue import Queue

# OpenPCDet imports
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_utils, common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset

# ROS imports
import rospy
import ros_numpy

# Math and geometry imports
import torch
import numpy as np

class Processor_ROS_nuScenes:
    
    def __init__(self, config_path, model_path):

        self.config_path = config_path
        self.dataset_cfg = None
        self.pcl_queue = None
        self.model_path = model_path
        self.model = None
        self.demo_dataset = None
        self.actual_time = 0

    def initilize_network(self):

        cfg_from_yaml_file(self.config_path, cfg)
        self.dataset_cfg = cfg.DATA_CONFIG
        self.pcl_queue = Queue(maxsize = self.dataset_cfg.MAX_SWEEPS) # (timestamp, pcl)

        logger = common_utils.create_logger()
        self.demo_dataset = NuScenesDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/none"), logger=logger
        )

        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=self.model_path, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

    def __get_xyzi_points(self, cloud_array):

        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

        points = np.zeros((len(cloud_array['x']),5))
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']
        points[...,3] = cloud_array['intensity']*0

        return points

    def __remove_ego_points(self, points, center_radius=2):

        mask = ~(np.vectorize(lambda x: abs(x[0]) < center_radius and abs(x[1]) < center_radius)(points))
        return points[mask]

    def new_pcl(self, msg):

        msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        msg_cloud = self.__remove_ego_points(msg_cloud)
        pcl = self.__get_xyzi_points(msg_cloud)
        timestap = msg.header.stamp.secs + msg.header.stamp.nsecs/1000000000 # Time in seconds
        self.actual_time = timestap
        if(self.pcl_queue.full()):
            self.pcl_queue.get()
        self.pcl_queue.put((timestap, pcl))

    def __get_pcl_sweeps(self):

        sweeps = self.pcl_queue.qsize()
        complete_pcl = []
        for _ in range(sweeps):
            pcl_elem = self.pcl_queue.get()
            time_difference = self.actual_time - pcl_elem[0]
            pcl = pcl_elem[1]
            pcl[:,4] = time_difference
            if complete_pcl != []:
                complete_pcl = np.concatenate((complete_pcl, pcl))
            else:
                complete_pcl = pcl
            self.pcl_queue.put(pcl_elem)
        return complete_pcl

    def inference(self):

        input_dict = {
            'points': self.__get_pcl_sweeps(),
            'frame_id': 0,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = self.model.forward(data_dict)

        return pred_dicts