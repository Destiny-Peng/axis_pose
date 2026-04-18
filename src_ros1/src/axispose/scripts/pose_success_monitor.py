#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image


class PoseSuccessMonitor:
    def __init__(self):
        self.window_sec = rospy.get_param("~window_sec", 10.0)
        self.mask_topic = rospy.get_param("~mask_topic", "/yolo/mask")
        self.pose_topic = rospy.get_param("~pose_topic", "/shaft/pose")

        self.mask_count = 0
        self.pose_count = 0
        self.last_mask_stamp = None
        self.last_pose_stamp = None

        rospy.Subscriber(self.mask_topic, Image, self.mask_cb, queue_size=20)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=20)

        self.timer = rospy.Timer(rospy.Duration(self.window_sec), self.report)
        rospy.loginfo("pose_success_monitor started: window=%.1fs mask=%s pose=%s", self.window_sec, self.mask_topic, self.pose_topic)

    def mask_cb(self, msg):
        self.mask_count += 1
        self.last_mask_stamp = msg.header.stamp

    def pose_cb(self, msg):
        self.pose_count += 1
        self.last_pose_stamp = msg.header.stamp

    def report(self, _):
        denom = max(1, self.mask_count)
        success_rate = float(self.pose_count) / float(denom)
        rospy.loginfo("window %.1fs: mask=%d pose=%d success_rate=%.3f",
                      self.window_sec, self.mask_count, self.pose_count, success_rate)

        self.mask_count = 0
        self.pose_count = 0


def main():
    rospy.init_node("pose_success_monitor")
    PoseSuccessMonitor()
    rospy.spin()


if __name__ == "__main__":
    main()
