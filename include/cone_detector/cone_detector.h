// Copyright 2017 Michael Wimble

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

#ifndef __VICTORIA_PERCEPTION_CONE_DETECTOR
#define __VICTORIA_PERCEPTION_CONE_DETECTOR
#include <ros/ros.h>
#include <ros/console.h>

#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

#include "victoria_perception/AnnotateDetectorImage.h"
#include "victoria_perception/CalibrateConeDetection.h"
#include "victoria_perception/ConeDetectorConfig.h"


class ConeDetector {
private:
	// Statics.
	static const int g_font_face = cv::FONT_HERSHEY_SIMPLEX;
	static const double g_font_scale;
	static const int g_font_line_thickness = 2;

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
	int alow_hue_range_;
	int ahigh_hue_range_;

	int alow_saturation_range_;
	int ahigh_saturation_range_;

	int alow_value_range_;
	int ahigh_value_range_;

	int blow_hue_range_;
	int bhigh_hue_range_;

	int blow_saturation_range_;
	int bhigh_saturation_range_;

	int blow_value_range_;
	int bhigh_value_range_;

	int low_contour_area_;
	int high_contour_area_;

	float max_aspect_ratio_;
	float poly_epsilon_;
	int erode_kernel_size_;
	// End values for the sample thresholding operation.

	bool debug_;

	// Name of camera, to get camera properties.
	std::string camera_name_;

	// Publish annotated image.
	image_transport::Publisher image_pub_annotated_;

	// Publish thresholded image.
	image_transport::Publisher image_pub_thresholded_;

	// Subscriber to images on "image_topic_name_" topic.
	image_transport::Subscriber image_sub_;

	// Topic publishing the image that might contain a traffic cone.
	std::string image_topic_name_;

	// OpenCV image transport.
	image_transport::ImageTransport it_;

	// Image will be resized to this x (cols) size.
	int resize_x_;

	// Image will be resized to this y (rows) size.
	int resize_y_;

	// Parameter-settable to show times taken for interesting steps in object recognition.
	bool show_step_times_;

	// Publisher handles.
	ros::Publisher cone_found_pub_;

	// Process AnnotateDetectorImage service call.
	ros::ServiceServer annotateService;
	std::string ll_annotation_;
	cv::Scalar ll_color_;
	std::string lr_annotation_;
	cv::Scalar lr_color_;
	std::string ul_annotation_;
	cv::Scalar ul_color_;
	std::string ur_annotation_;
	cv::Scalar ur_color_;

	// Process CalibrateConeDetection service call.
	ros::ServiceServer calibrateConeDetectionService;

	// Handle annotation service.
	bool annotateCb(victoria_perception::AnnotateDetectorImage::Request &request,
			victoria_perception::AnnotateDetectorImage::Response &response);

	// Handle calibrate cone detection service.
	bool calibrateConeDetectionCb(victoria_perception::CalibrateConeDetection::Request &request,
			victoria_perception::CalibrateConeDetection::Response &response);

	// Dynamic reconfiguration.
	dynamic_reconfigure::Server<victoria_perception::ConeDetectorConfig> dynamic_server_;
	dynamic_reconfigure::Server<victoria_perception::ConeDetectorConfig>::CallbackType configCallbackType_;
	void configCb(victoria_perception::ConeDetectorConfig &config, uint32_t level);

	// Convert a 6-digit hexadecimal string into a blue-green-red color.
	static bool strToBgr(std::string bgr_string, cv::Scalar& out_color);
	
	bool hullIsValid(std::vector<cv::Point>& hull);

	// Process one image.
	cv::Mat last_image_;
	long last_image_count_;
	void imageCb(cv::Mat& image);

	// Process one image topic message.
	void imageTopicCb(const sensor_msgs::ImageConstPtr& msg);

	// Compute KMEANS on image.
	void kmeansImage(cv::Mat image);

	// Put requested annotations in image.
	void placeAnnotationsInImage(cv::Mat annotation_image) ;

	// singleton pattern.
	ConeDetector();
	ConeDetector(ConeDetector const&) : it_(nh_) {};
	ConeDetector& operator=(ConeDetector const&) {};

public:
	// Obtain the single instance of this class.
	static ConeDetector& singleton();

	int imageHeight() { return image_height_; }
	int imageWidth() { return image_width_; }
	int objectArea() { return object_area_; }	// Square pixel area of detected object.
	bool objectDetected() { return object_detected_; }
	int objectX() { return object_x_; }
	int objectY() { return object_y_; }
};

#endif  // __VICTORIA_PERCEPTION_CONE_DETECTOR
