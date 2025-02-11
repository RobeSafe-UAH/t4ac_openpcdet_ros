# General imports

import struct
import ctypes
import rospy

# ROS imports

import rospy
from visualization_msgs.msg import Marker
from t4ac_msgs.msg import Bounding_Box_3D, Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg

# Math and geometry imports

import math
import torch
import numpy as np
import torchvision.ops.boxes as bops

# Auxiliar functions/classes imports

from modules.auxiliar_functions import yaw2quaternion, calculate_3d_corners

# Object types

classes = ["Car",
           "Truck",
           "Construction_Vehicle",
           "Bus",
           "Trailer",
           "Barrier",
           "Motorcycle",
           "Bicycle",
           "Pedestrian",
           "Traffic_Cone"]

def get_bounding_box_3d(box, score, label):

    bounding_box_3d = Bounding_Box_3D()

    bounding_box_3d.type = classes[label-1]
    bounding_box_3d.score = score

    bounding_box_3d.pose.pose.position.x = box[0]
    bounding_box_3d.pose.pose.position.y = box[1]
    bounding_box_3d.pose.pose.position.z = box[2]

    q = yaw2quaternion(box[6])
    bounding_box_3d.pose.pose.orientation.x = q[1] 
    bounding_box_3d.pose.pose.orientation.y = q[2]
    bounding_box_3d.pose.pose.orientation.z = q[3]
    bounding_box_3d.pose.pose.orientation.w = q[0]

    bounding_box_3d.l = box[3]
    bounding_box_3d.w = box[4]
    bounding_box_3d.h = box[5]

    bounding_box_3d.corners_3d = calculate_3d_corners(box)

    bounding_box_3d.vel_lin = math.sqrt(pow(box[7],2)+pow(box[8],2))

    return bounding_box_3d

def marker_bb(header,box,label,id_marker,corners=False):
    """
    If corners = True, visualize the 3D corners instead of a solid cube
    """
    colors_label = [(0,0,255), (255,255,0), (128,0,0), # car(blue), truck(yellow), construction_vehicle(maroon)
                    (0,128,128), (0,128,0), (0,255,255), # bus(teal), trailer(green), barrier(cyan)
                    (0,255,0), (128,128,128), (255,0,255), (128,0,128)] #motorcycle(lime), bicycle(grey), pedestrian(magenta), traffic_cone(purple)

    box_marker = Marker()
    box_marker.header.stamp = header.stamp
    box_marker.header.frame_id = header.frame_id
    box_marker.action = Marker.ADD
    box_marker.id = id_marker
    box_marker.lifetime = rospy.Duration.from_sec(0.1)
    box_marker.ns = "multihead_obstacles"

    if corners:
        box_marker.type = Marker.POINTS
        box_marker.scale.x = 0.3
        box_marker.scale.y = 0.3
        box_marker.scale.z = 0.3
        box_marker.pose.orientation.w = 1.0

        corners_3d = calculate_3d_corners(box)

        for corner in corners_3d:
            pt = Point()

            pt.x = corner.x
            pt.y = corner.y
            pt.z = corner.z

            box_marker.points.append(pt)

        color_norm = map(lambda x: x/255, colors_label[label-1])
        box_marker.color.r, box_marker.color.g, box_marker.color.b = color_norm
        box_marker.color.a = 1.0

        return box_marker
    else:
        box_marker.type = Marker.CUBE
        box_marker.pose.position.x = box[0]
        box_marker.pose.position.y = box[1]
        box_marker.pose.position.z = box[2]
        q = yaw2quaternion(box[6])
        box_marker.pose.orientation.x = q[1] 
        box_marker.pose.orientation.y = q[2]
        box_marker.pose.orientation.z = q[3]
        box_marker.pose.orientation.w = q[0]
        box_marker.scale.x = box[3]
        box_marker.scale.y = box[4]
        box_marker.scale.z = box[5]
        color_norm = map(lambda x: x/255, colors_label[label-1])
        box_marker.color.r, box_marker.color.g, box_marker.color.b = color_norm
        box_marker.color.a = 0.5

        return box_marker

def marker_arrow(header, box, label, id_marker):
    """
    """
    arrow_marker = Marker()
    arrow_marker.header.stamp = header.stamp
    arrow_marker.header.frame_id = header.frame_id
    arrow_marker.type = Marker.ARROW
    arrow_marker.action = Marker.ADD
    arrow_marker.id = id_marker
    arrow_marker.lifetime = rospy.Duration.from_sec(1)
    arrow_marker.ns = "multihead_velocities"
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
    p1.x = box[0]+box[7]
    p1.y = box[1]+box[8]
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

def relative2absolute_velocity(pred_boxes, ego_vel_x_local, ego_vel_y_local):

    pred_boxes[:,7] += msg_odometry.twist.twist.linear.x
    pred_boxes[:,7] += ego_vel_x_local

    return pred_boxes

def relative2absolute_velocity(pred_boxes, msg_odometry):
    pred_boxes[:,7] += msg_odometry.twist.twist.linear.x
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