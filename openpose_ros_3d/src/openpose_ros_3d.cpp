// ------------------------- OpenPose Library Tutorial - Real Time Pose Estimation -------------------------

// C++ std library dependencies
#include <atomic>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <cstdio> // sscanf
#include <cstdlib>
#include <string>
#include <vector>
// Other 3rdpary depencencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string

// OpenPose dependencies
// Option a) Importing all modules
#include <openpose/headers.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <openpose_ros_msgs/Persons.h>
#include <openpose_ros_msgs/BodyPartDetection.h>
#include <openpose_ros_msgs/PersonDetection.h>
#include <openpose_ros_msgs/Persons_3d.h>
#include <openpose_ros_msgs/BodyPartDetection_3d.h>
#include <openpose_ros_msgs/PersonDetection_3d.h>
#include "openpose_ros_common.hpp"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

//body connections of the coco model
const int connections[17][2] = {{0,1},{1,2},{2,3},{3,4},{1,5},{5,6},{6,7},{1,8},{8,9},{9,10},{1,11},{11,12},{12,13},{0,14},{0,15},{14,16},{15,17}};

openpose_ros_msgs::BodyPartDetection_3d bodypart_3d_init(int part_id){
  openpose_ros_msgs::BodyPartDetection_3d newBodypart;
  newBodypart.part_id = part_id;
  newBodypart.x = 0;
  newBodypart.y = 0;
  newBodypart.z = 0;
  newBodypart.confidence = 0.0f;
  return newBodypart;
}

geometry_msgs::Point addPoint(const openpose_ros_msgs::BodyPartDetection_3d bodypart){
  geometry_msgs::Point p;
  p.x = bodypart.x;
  p.y = bodypart.y;
  p.z = bodypart.z;
  return p;
}

bool pointIsValid(pcl::PointXYZ p){
    return(!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z));
}

std::vector<double> pointsInCircle(pcl::PointCloud<pcl::PointXYZ> depthCloud, int radius, int centerX, int centerY, int width, int height){
  std::vector<double> circlePoints;
  //check in the first quadrant if the is in the circle and then add the symmetric points on the other quadrants

  for(int x = centerX; x <= centerX + radius; x++){
    for(int y = centerY; y <= centerY + radius; y++){
      int otherX = centerX-(x-centerX);
      int otherY = centerY-(y-centerY);
      //if point (x,y) is in the circle
      if((x-centerX)*(x-centerX) + (y-centerY)*(y-centerY) <= radius*radius){
        if(x < width && y < height){

          circlePoints.push_back(depthCloud.points[640*y+x].z);
        }
        if(x < width && otherY >= 0){

          circlePoints.push_back(depthCloud.points[640*otherY+x].z);
        }
        if(y < width && otherX >= 0){

          circlePoints.push_back(depthCloud.points[640*y+otherX].z);
        }
        if(otherX >= 0 && otherY >= 0){

          circlePoints.push_back(depthCloud.points[640*otherY+otherX].z);
        }
      }
    }
  }
  return circlePoints;
}

double median(std::vector<double> v){
  size_t size = v.size();
  std::sort(v.begin(),v.end());
  if(size % 2 == 0)
    return((v[size/2 - 1] + v[size/2])/2);
  else
    return(v[size/2]);
}

double average(std::vector<double> v){
  double total = 0;
  for(int i = 0; i < v.size(); i++){
    total += v[i];
  }
  return(total/v.size());
}

class OpenPoseRos3d
{
private:
  ros::NodeHandle nh;
  ros::Subscriber cloud_sub;
  ros::Subscriber keypoints2d_sub;
  ros::Publisher keypoints3d_pub;
  ros::Publisher points_pub;
  ros::Publisher joints_pub;
  pcl::PointCloud<pcl::PointXYZ> depthCloud;
public:
  OpenPoseRos3d(){
    keypoints3d_pub =nh.advertise<openpose_ros_msgs::Persons_3d>("keypoints3d", 1);
    points_pub =nh.advertise<visualization_msgs::Marker>("markers", 1);
    joints_pub =nh.advertise<visualization_msgs::Marker>("joints", 1);
    cloud_sub = nh.subscribe("/camera/depth/points", 1, &OpenPoseRos3d::cloudCallback, this);
    keypoints2d_sub = nh.subscribe("/openpose/pose", 1, &OpenPoseRos3d::keypointsCallback, this);
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
    //update current point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *temp_cloud);
    depthCloud = *temp_cloud;
  }

  void keypointsCallback(const openpose_ros_msgs::Persons::ConstPtr& keypoints2d_msg){
    openpose_ros_msgs::Persons_3d keypoints3d_msg;
    keypoints3d_msg.header.stamp = ros::Time::now();
    keypoints3d_msg.image_w = 640;
    keypoints3d_msg.image_h = 480;

    visualization_msgs::Marker marker;
    marker.header.frame_id = "/camera_depth_frame";
    marker.id = 0;
    marker.ns = "joints";
    marker.header.stamp = ros::Time();
    // Markers will be spheres
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    // Joints are red
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.lifetime = ros::Duration(0.5);

    visualization_msgs::Marker skeleton;
    skeleton.id = 0;
    skeleton.header.frame_id = "/camera_depth_frame";
    skeleton.ns = "skeleton";
    skeleton.header.stamp = ros::Time();
    // Skeleton will be lines
    skeleton.type = visualization_msgs::Marker::LINE_LIST;
    skeleton.scale.x = 0.03;
    skeleton.scale.y = 0.03;
    skeleton.scale.z = 0.03;
    // Skeleton is blue
    skeleton.color.a = 1.0;
    skeleton.color.r = 0.0;
    skeleton.color.g = 0.0;
    skeleton.color.b = 1.0;
    skeleton.lifetime = ros::Duration(0.5);
    //get 3d coordinates of the keypoints
    for(int i = 0; i < keypoints2d_msg->persons.size(); i++)
    {
      openpose_ros_msgs::PersonDetection_3d person;
      for(int j = 0; j < keypoints2d_msg->persons[i].body_part.size(); j++)
      {
        openpose_ros_msgs::BodyPartDetection_3d bodypart = bodypart_3d_init(j);
        int x = keypoints2d_msg->persons[i].body_part[j].x;
        int y = keypoints2d_msg->persons[i].body_part[j].y;
        if(pointIsValid(depthCloud.points[640*y+x]) && keypoints2d_msg->persons[i].body_part[j].confidence > 0.2f)
        {
          bodypart.confidence = keypoints2d_msg->persons[i].body_part[j].confidence;
          bodypart.x = depthCloud.points[640*y+x].x;
          bodypart.y = depthCloud.points[640*y+x].y;
          bodypart.z = median(pointsInCircle(depthCloud, 3, x, y, 640, 480));
        }
        person.body_part.push_back(bodypart);
      }
      keypoints3d_msg.persons.push_back(person);
    }
    keypoints3d_pub.publish(keypoints3d_msg);

    for(int i = 0; i < keypoints3d_msg.persons.size(); i++)
    {
      //add keypoints to the marker
      bool part[18];
      for(int j = 0; j < keypoints3d_msg.persons[i].body_part.size(); j++)
      {
        if(keypoints3d_msg.persons[i].body_part[j].z < 2 && keypoints3d_msg.persons[i].body_part[j].confidence > 0)
        {
          marker.points.push_back(addPoint(keypoints3d_msg.persons[i].body_part[j]));
          part[j] = true;
        }
        else part[j] = false;
      }
      //connect keypoints
      for(int j = 0; j < 17; j++){
        int a = connections[j][0];
        int b = connections[j][1];
        if(part[a] && part[b]){
          skeleton.points.push_back(addPoint(keypoints3d_msg.persons[i].body_part[a]));
          skeleton.points.push_back(addPoint(keypoints3d_msg.persons[i].body_part[b]));
        }
      }
    }
    points_pub.publish(marker);
    joints_pub.publish(skeleton);
  }
};

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "openpose_ros_3d");
  OpenPoseRos3d openposeRos3d;

  /*message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, "/camera/depth/points", 1);
  message_filters::Subscriber<openpose_ros_msgs::Persons> keypoints2d_sub(nh, "/openpose/pose", 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, openpose_ros_msgs::Persons> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), cloud_sub, keypoints2d_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));*/

  ros::spin();
  return 0;
}
