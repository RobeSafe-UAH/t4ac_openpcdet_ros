# General imports
import struct
import ctypes
import rospy

# ROS imports
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg

# Math and geometry imports
import math
import torch
import numpy as np
import torchvision.ops.boxes as bops

# Auxiliar functions/classes imports
from modules.auxiliar_functions import yaw2quaternion

def marker_bb(msg, box, score, label, id_marker):

    colors_label = [(0,0,255), (255,255,0), (128,0,0), # car(blue), truck(yellow), construction_vehicle(maroon)
                    (0,128,128), (0,128,0), (0,255,255), # bus(teal), trailer(green), barrier(cyan)
                    (0,255,0), (128,128,128), (255,0,255), (128,0,128)] #motorcycle(lime), bicycle(grey), pedestrian(magenta), traffic_cone(purple)

    box_marker = Marker()
    box_marker.header.stamp = msg.header.stamp
    box_marker.header.frame_id = msg.header.frame_id
    box_marker.type = Marker.CUBE
    box_marker.id = id_marker
    box_marker.lifetime = rospy.Duration.from_sec(1)
    box_marker.pose.position.x = box[0]
    box_marker.pose.position.y = box[1]
    box_marker.pose.position.z = box[2]
    box_marker.scale.x = box[3]
    box_marker.scale.y = box[4]
    box_marker.scale.z = box[5]
    quaternion = yaw2quaternion(box[6])
    box_marker.pose.orientation.x = quaternion[1] 
    box_marker.pose.orientation.y = quaternion[2]
    box_marker.pose.orientation.z = quaternion[3]
    box_marker.pose.orientation.w = quaternion[0]
    box_marker.color.r, box_marker.color.g, box_marker.color.b = colors_label[label-1]
    box_marker.color.a = 0.3

    return box_marker

def marker_arrow(msg, box, score, label, id_marker):

    arrow_marker = Marker()
    arrow_marker.header.stamp = msg.header.stamp
    arrow_marker.header.frame_id = msg.header.frame_id
    arrow_marker.type = Marker.ARROW
    arrow_marker.id = id_marker
    arrow_marker.lifetime = rospy.Duration.from_sec(1)
    arrow_marker.color.r = 255
    arrow_marker.color.g = 255
    arrow_marker.color.b = 255
    arrow_marker.color.a = 1
    arrow_marker.pose.orientation.x = 0
    arrow_marker.pose.orientation.y = 0
    arrow_marker.pose.orientation.z = 0
    arrow_marker.pose.orientation.w = 1
    arrow_marker.scale.x = 0.2
    arrow_marker.scale.y = 0.2
    arrow_marker.scale.z = 0.2
        
    p0 = Point()
    p1 = Point()
    p0.x = box[0]
    p0.y = box[1]
    p0.z = box[2]
    p1.x = box[0]-box[7]
    p1.y = box[1]-box[8]
    p1.z = box[2]
    arrow_marker.points = [p0, p1]

    return arrow_marker

def calculate_iou(box_1, box_2):

    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

    return iou

def filter_predictions(pred_dicts, simulation):
    
    confidence_threshold = [0.35, 0.5, 0.45, # car, truck, construction_vehicle
                            0.5, 0.45, 0.5, # bus, trailer, barrier
                            0.3, 0.2, 0.05, 0.05] #motorcycle, bicycle, pedestrian, traffic_cone

    pred_boxes, pred_scores, pred_labels = (pred_dicts[0]['pred_boxes'].cpu().detach().numpy(),
                                            pred_dicts[0]['pred_scores'].cpu().detach().numpy(),
                                            pred_dicts[0]['pred_labels'].cpu().detach().numpy())

    # Mask to filter by the threshold
    if(len(pred_boxes) != 0):
        indices = np.vectorize(lambda score, label: score > confidence_threshold[label-1])(pred_scores, pred_labels)
    
        pred_boxes = pred_boxes[indices].reshape(-1, 9)
        pred_scores = pred_scores[indices].reshape(-1)
        pred_labels = pred_labels[indices].reshape(-1)

    # Sort by the score of the detection
    if(len(pred_boxes) != 0):
        sorted_tuple = sorted(zip(pred_scores, pred_boxes, pred_labels))
        pred_scores, pred_boxes, pred_labels = ([a.tolist() for a,b,c in sorted_tuple], np.array([b for a,b,c in sorted_tuple]), [c for a,b,c in sorted_tuple])

    # Get the detentions that are different from the rest based on iou 2D
    treshold_iou_3d = 0.5
    n_detections = len(pred_boxes)
    indices = [True] * n_detections
    for i in range(n_detections):
        box0 = pred_boxes[i]
        box0 = torch.tensor([[box0[0]-box0[3]/2, box0[1]-box0[4]/2, box0[0]+box0[3]/2, box0[1]+box0[4]/2]], dtype=torch.float)
        for j in range(i+1,n_detections):
            box1 = pred_boxes[j]
            box1 = torch.tensor([[box1[0]-box1[3]/2, box1[1]-box1[4]/2, box1[0]+box1[3]/2, box1[1]+box1[4]/2]], dtype=torch.float)
            if(treshold_iou_3d < bops.box_iou(box0, box1)):
                indices[i] = False

    return pred_boxes, pred_scores, pred_labels

def relative2absolute_velocity(pred_boxes, msg_odometry):

    pred_boxes[:,7] += msg_odometry.twist.twist.linear.x * np.vectorize((lambda x: math.sin(x)))(pred_boxes[:,6]) - msg_odometry.twist.twist.linear.y * np.vectorize((lambda x: math.cos(x)))(pred_boxes[:,6])
    pred_boxes[:,8] += - msg_odometry.twist.twist.linear.y * np.vectorize((lambda x: math.cos(x)))(pred_boxes[:,6]) - msg_odometry.twist.twist.linear.y * np.vectorize((lambda x: math.sin(x)))(pred_boxes[:,6])

    return pred_boxes

def publish_pcl2(pub_pcl2, points):

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1)]
    _DATATYPES = {}
    _DATATYPES[PointField.INT8]    = ('b', 1)
    _DATATYPES[PointField.UINT8]   = ('B', 1)
    _DATATYPES[PointField.INT16]   = ('h', 2)
    _DATATYPES[PointField.UINT16]  = ('H', 2)
    _DATATYPES[PointField.INT32]   = ('i', 4)
    _DATATYPES[PointField.UINT32]  = ('I', 4)
    _DATATYPES[PointField.FLOAT32] = ('f', 4)
    _DATATYPES[PointField.FLOAT64] = ('d', 8)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.seq = 0
    header.frame_id = 'map'
    def _get_struct_fmt(is_bigendian, fields, field_names=None):
        fmt = '>' if is_bigendian else '<'
        offset = 0
        for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
            if offset < field.offset:
                fmt += 'x' * (field.offset - offset)
                offset = field.offset
            if field.datatype not in _DATATYPES:
                print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
            else:
                datatype_fmt, datatype_length = _DATATYPES[field.datatype]
                fmt    += field.count * datatype_fmt
                offset += field.count * datatype_length
        return fmt
    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))
    buff = ctypes.create_string_buffer(cloud_struct.size * len(points))
    point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
    offset = 0
    for p in points:
        pack_into(buff, offset, *p)
        offset += point_step
    pcl2 = PointCloud2(header=header, height=1, width=len(points), is_dense=False, is_bigendian=False,
                        fields=fields, point_step=cloud_struct.size,
                        row_step=cloud_struct.size * len(points), data=buff.raw)
    pub_pcl2.publish(pcl2)