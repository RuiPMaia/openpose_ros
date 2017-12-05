#include <atomic>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <cstdlib>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <openpose_ros_msgs/Persons.h>
#include <openpose_ros_msgs/BodyPartDetection.h>
#include <openpose_ros_msgs/PersonDetection.h>
#include <openpose_ros_msgs/BoundingBox.h>
#include <openpose_ros_msgs/BBPerson.h>
#include <openpose_ros_msgs/BBList.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, openpose_ros_msgs::Persons> MySyncPolicy;

const std::vector<int> headPoints = {0, 14, 15, 16, 17};
const std::vector<int> torsoPoints = {1, 8, 11};
const std::vector<int> legPoints = {8, 9, 11, 12};
const std::vector<int> feetPoints = {9, 10, 12, 13};

std::string image_src, result_image_topic;
double keypoints_threshold;

openpose_ros_msgs::BoundingBox calculatePartMask(std::vector<openpose_ros_msgs::BodyPartDetection> bodyParts, const std::vector<int> maskPoints)
{
	openpose_ros_msgs::BoundingBox box;
	int minX = bodyParts[maskPoints[0]].x;
	int maxX = bodyParts[maskPoints[0]].x;
	int minY = bodyParts[maskPoints[0]].y;
	int maxY = bodyParts[maskPoints[0]].y;
	for(int i = 1; i < maskPoints.size(); i++){
		int idx = maskPoints[i];
		if(bodyParts[idx].confidence == 0) continue;
		if(bodyParts[idx].x > maxX) maxX = bodyParts[idx].x;
		if(bodyParts[idx].x < minX) minX = bodyParts[idx].x;
		if(bodyParts[idx].y > maxY) maxY = bodyParts[idx].y;
		if(bodyParts[idx].y < minY) minY = bodyParts[idx].y;
	}
	box.x = minX;
	box.y = minY;
	box.width = maxX-minX;
	box.height = maxY-minY;
	return box;
}

class bboxCreator
{
private:
	ros::NodeHandle nh;
	image_transport::ImageTransport it;
	image_transport::Publisher image_pub;
	ros::Publisher bbox_pub;
	message_filters::Subscriber<sensor_msgs::Image> image_sub;
	message_filters::Subscriber<openpose_ros_msgs::Persons> keypoints_sub;
	message_filters::Synchronizer< MySyncPolicy > sync;
public:
 	bboxCreator():
  		it(nh),
		image_sub(nh, image_src, 1),
		keypoints_sub(nh, "/openpose/pose", 1),
		sync(MySyncPolicy(10), image_sub, keypoints_sub)
		{
			image_pub = it.advertise(result_image_topic, 1);
			bbox_pub =nh.advertise<openpose_ros_msgs::BBList>("openpose/bounding_box", 1);
			sync.registerCallback(boost::bind(&bboxCreator::callback, this, _1, _2));
		}

	void callback(const sensor_msgs::ImageConstPtr& img_msg, const openpose_ros_msgs::Persons::ConstPtr& keypoints2d_msg)
	{
		cv_bridge::CvImagePtr cv_ptr;
	 	try {
	 		cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
	 	}
		catch (cv_bridge::Exception& e){
			return;
		}
	 	if (cv_ptr->image.empty()) return;

		openpose_ros_msgs::BBList bbList;

		for(int i = 0; i < keypoints2d_msg->persons.size(); i++){
			openpose_ros_msgs::PersonDetection person = keypoints2d_msg->persons[i];
			//if the torso isn't visible don create a bounding box
			if(person.body_part[1].confidence < keypoints_threshold || person.body_part[8].confidence < keypoints_threshold ||
				person.body_part[11].confidence < keypoints_threshold) continue;

			//consider only keypoints above a certain threshold
			std::vector<openpose_ros_msgs::BodyPartDetection> bodyParts;
			for(int j = 0; j < person.body_part.size(); j++){
				if(person.body_part[j].confidence >= keypoints_threshold)
					bodyParts.push_back(person.body_part[j]);
			}
			//calculate min and max values for x and y coordinates of the keypoints
			int minX = bodyParts[0].x;
			int maxX = bodyParts[0].x;
			int minY = bodyParts[0].y;
			int maxY = bodyParts[0].y;
			for(int j = 1; j < bodyParts.size(); j++){
				if(bodyParts[j].x > maxX) maxX = bodyParts[j].x;
				if(bodyParts[j].x < minX) minX = bodyParts[j].x;
				if(bodyParts[j].y > maxY) maxY = bodyParts[j].y;
				if(bodyParts[j].y < minY) minY = bodyParts[j].y;
			}
			//if height or width are 0 don't create a Bounding box
			//if(maxX-minX == 0 || maxY-minY == 0) continue;
			openpose_ros_msgs::BBPerson bbPerson;
			//calculate part masks
			bbPerson.partMasks.push_back(calculatePartMask(person.body_part, headPoints));
			bbPerson.partMasks.push_back(calculatePartMask(person.body_part, torsoPoints));
			bbPerson.partMasks.push_back(calculatePartMask(person.body_part, legPoints));
			bbPerson.partMasks.push_back(calculatePartMask(person.body_part, feetPoints));
			//adjustments to the masks
			int dist = person.body_part[1].y - person.body_part[0].y; //vertical distance from nose to neck
			bbPerson.partMasks[0].height += dist/2; //adjust head mask
			bbPerson.partMasks[3].height *= 1.5; //adjust feet mask
			//average torso length
			int ref = (person.body_part[8].y+person.body_part[11].y)/2-person.body_part[1].y;
			bbPerson.bbox.x = minX - ref*0.15;
			bbPerson.bbox.y = minY - ref*0.25;
			bbPerson.bbox.width = maxX + ref*0.15 - bbPerson.bbox.x;
			bbPerson.bbox.height = maxY + ref*0.35 - bbPerson.bbox.y ;
			bbList.bbPersons.push_back(bbPerson);
			cv::rectangle(cv_ptr->image, cv::Rect(bbPerson.bbox.x, bbPerson.bbox.y, bbPerson.bbox.width, bbPerson.bbox.height), cv::Scalar(0, 255, 0));
			cv::rectangle(cv_ptr->image, cv::Rect(bbPerson.partMasks[0].x, bbPerson.partMasks[0].y, bbPerson.partMasks[0].width, bbPerson.partMasks[0].height), cv::Scalar(255, 0, 0));
			cv::rectangle(cv_ptr->image, cv::Rect(bbPerson.partMasks[1].x, bbPerson.partMasks[1].y, bbPerson.partMasks[1].width, bbPerson.partMasks[1].height), cv::Scalar(255, 0, 0));
			cv::rectangle(cv_ptr->image, cv::Rect(bbPerson.partMasks[2].x, bbPerson.partMasks[2].y, bbPerson.partMasks[2].width, bbPerson.partMasks[2].height), cv::Scalar(255, 0, 0));
			cv::rectangle(cv_ptr->image, cv::Rect(bbPerson.partMasks[3].x, bbPerson.partMasks[3].y, bbPerson.partMasks[3].width, bbPerson.partMasks[3].height), cv::Scalar(255, 0, 0));
		}

		bbox_pub.publish(bbList);
		image_pub.publish(cv_ptr->toImageMsg());
	}
};

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "openpose_ros_bbox");
	ros::NodeHandle local_nh("~");
	local_nh.getParam("image_src", image_src);
	local_nh.getParam("result_image_topic", result_image_topic);
	local_nh.getParam("keypoints_threshold", keypoints_threshold);

	bboxCreator bbox;
	ros::spin();
	return 0;
}
