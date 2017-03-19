// Copyright <YEAR> <COPYRIGHT HOLDER>

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
// or promote products derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __CONE_DETECTOR
#define __CONE_DETECTOR
#include <ros/ros.h>
#include <ros/console.h>

#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include "opencv2/core/core.hpp"
#include <string>

#include "victoria_perception/AnnotateDetectorImage.h"


using namespace std;
using namespace cv;

class ConeDetector {
private:
	// Detector products.
	int image_height_;
	int image_width_;
	int object_area_;
	bool object_detected_;
	int object_x_;
	int object_y_;

	// ROS node handle.
	ros::NodeHandle nh_;

	// HSV Values and contour area range for the sample thresholding operation.
	int low_hue_range_;
	int high_hue_range_;

	int low_saturation_range_;
	int high_saturation_range_;

	int low_value_range_;
	int high_value_range_;

	int low_contour_area_;
	int high_contour_area_;
	// End values for the sample thresholding operation.

	// Name of camera, to get camera properties.
	string camera_name_;

	// Publish annotated image.
	image_transport::Publisher image_pub_annotated_;

	// Publish thresholded image.
	image_transport::Publisher image_pub_thresholded_;

	// Subscriber to images on "image_topic_name_" topic.
	image_transport::Subscriber image_sub_;

	// Topic publishing the image that might contain a traffic cone.
	string image_topic_name_;

	// OpenCV image transport.
	image_transport::ImageTransport it_;

	// Image will be resized to this x (cols) size.
	int resize_x_;

	// Image will be resized to this y (rows) size.
	int resize_y_;

	// Parameter-settable to show X-windows debugging windows.
	bool show_debug_windows_;

	// Parameter-settable to show times taken for interesting steps in object recognition.
	bool show_step_times_;

	// Publisher handles.
	ros::Publisher cone_found_pub_;

//	dynamic_reconfigure::Server<kaimi_mid_camera::kaimi_mid_camera_paramsConfig> dynamicConfigurationServer;
//	dynamic_reconfigure::Server<kaimi_mid_camera::kaimi_mid_camera_paramsConfig>::CallbackType f;
//
//	static void configurationCallback(kaimi_mid_camera::kaimi_mid_camera_paramsConfig &config, uint32_t level);

	// Process service call.
	ros::ServiceServer annotateService;
	string ll_annotation_;
	Scalar ll_color_;
	string lr_annotation_;
	Scalar lr_color_;
	string ul_annotation_;
	Scalar ul_color_;
	string ur_annotation_;
	Scalar ur_color_;
	bool annotateCb(victoria_perception::AnnotateDetectorImage::Request &request,
			victoria_perception::AnnotateDetectorImage::Response &response);

	static bool strToBgr(string bgr_string, Scalar& out_color);
	
	// Process one image.
	void imageCb(Mat& image);

	// Process one image topic message.
	void imageTopicCb(const sensor_msgs::ImageConstPtr& msg);

	// singleton pattern.
	ConeDetector();
	ConeDetector(ConeDetector const&) : it_(nh_) {};
	ConeDetector& operator=(ConeDetector const&) {};

public:
	// Obtain the single instance of this class.
	static ConeDetector& singleton();

	int imageHeight() { return image_height_; }
	int imageWidth() { return image_width_; }
	int objectArea() { return object_area_; }
	bool objectDetected() { return object_detected_; }
	int objectX() { return object_x_; }
	int objectY() { return object_y_; }
};

#endif
