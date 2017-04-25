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

#ifndef __VICTORIA_PERCEPTION_KMEANS
#define __VICTORIA_PERCEPTION_KMEANS

#include <ros/ros.h>
#include <ros/console.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

#include "victoria_perception/ComputeKmeans.h"

//* A KmeansService class.
/**
* This is a ROS service handler for the compute_kmeans service. The service request takes as parameters:
* (see ComputeKmeans.srv)
* int16 	attempts 			The 'attempts' parameter to the cv::kmeans method.
* string 	image_topic_name 	The topic name of the video stream.
* int16 	number_clusters 	The 'K' parameter of cv::kmeans.
* int16 	resize_width 		The downsample width of the original image to work with. It must be <= original image width.
* bool 	show_annotated_window 	True => pop up an openCV window that is colorized to show the clusters.
*
* It responds with two fields:
* string 	result_msg 			"OK" for good result, else an error message.
* string 	kmeans_result		The kmeans results.
*
* The kmeans_result, if result_msg is "OK" is formatted as a sequence of cluster results. Each cluster result begins
* with a left curly brace and ends with a right curly brace. There are no characters between the right curly brace of
* one cluser result and the left curly brace of the next.
*
* Within the curly braces of a cluster result are a sequence of name, equal sign, value triples separated by semicolons.
*
* Here is a snippet of a kmeans_result string:
*
* {cluster=0;min_hue=15;max_hue=104;min_saturation=8;max_saturation=60;min_value=189;max_value=216;pixels=6635}{cluster=1;...}...
*
* The service constructs a listener for the next frame in the video stream for the topic, does the computation, then
* uses the normal destructors to remove the listener.
*/
class KmeansService {
private:
	/*!
	* \brief Holds the value range (min, max) and pxiels in a cluster for one channel in
	* a cluster (viz., hue channel, saturation channel or value channel).
	*/
	typedef struct ChannelRange {
	    int min;      // Min channel value.
	    int max;      // Max channel value.
	    int count;      // Number of pixels in the cluster.
	} ChannelRange;

	/*!
	* Indicates success or failure of a function.
	*/
	enum RESULT_T {
		SUCCESS,
		ERROR
	};

	// ROS node handles.
	ros::NodeHandle nh_;						// Main ROS node handle.
	ros::ServiceServer compute_kmeans_service_;	// Handle to accept service requests.

	// Parameters
	int attempts_;					// cv::kmeans 'attempts' value.
	std::string image_topic_name_;	// Topic name containing video stream.
	int number_clusters_;			// Number of clusters to form (cv::kmeans 'K' value).
	int resize_width_;				// Downsample image to this width (height will be proportional).
	bool show_annotated_window_;	// True => pop up window with posterized image where each poster color corresponds to a cluster.

	std::string result_set_;		// The string response to be returned on success.

	/*! \brief Popup an OpenCV window of the downsampled image where each pixel
	* is replaced by the random color associated with a cluster. On the right
	* of the image, a sequence of colored boxes will be drawn. The topmost
	* box corresponds to cluster 0 and is the color of all the false colored
	* pixels in teh annotated image for cluster 0. The box below corresponds
	* to the color of pixels for cluster 1 and so forth.
	* \param image 	The downsampled image which is copied and annotated.
	*/
	void createAnnotatedImage(const cv::Mat &image, std::vector<cv::Vec3b> clusters);

	/*! \brief Create the result string for the service call.
	* \param image 	The downsampled image. Used to find the colors of all pixels in a cluster.
	*/
	void createResultSet(const cv::Mat& image);

	/*! \brief Downsample the original image for faster computation.
	* \param original_image 	The original image.
	* \param out_image 			A copy of the original image that has been downsampled.
	*/
	void downSampleImage(const cv::Mat &original_image, cv::Mat& out_image);

	/*! \brief Get the min/max value for a given channel in a give cluster.
	* Note that the algorihm computes the pixel count of all possible values for the given channel in the cluster.
	* Hue values range in [0..179] and both saturation and value range in [0..255]. Because kmeans tends to include
	* pixels in a cluster that doesn't seem to be well matched for any given channel, after the counts are totaled,
	* a set of small count values are trimmed from both ends of the span and it's the reduced value range that is returned.
	* \param image 			The downsampled image.
	* \param cluster_index 	Which cluster to provide statistic for.
	* \param channel_index	Which channel to provide statistic for. 0=>Hue, 1=>Saturation, 2=>Value.
	*/
	ChannelRange getClusterStatistics(const cv::Mat &image, int cluster_index, int channel_index);
	
	/*! \brief Get the min/max of hue, saturation and value for all pixels in a cluster, as well as
	* the count of all pixels in that cluster. See also getClusterStatistics for a note on the reduced range returned.
	* \param image 				The downsampled image.
	* \param labels 			The kmeans label set. Each label corresponds to a pixel and the value is the cluster index for that pixel.
	* \param min_hue			Out - the minimum hue of all pixels in the cluster.
	* \param max_hue			Out - the maximum hue of all pixels in the cluster.
	* \param min_saturation		Out - the minimum saturation of all pixels in the cluster.
	* \param max_saturation		Out - the maximum saturation of all pixels in the cluster.
	* \param min_value			Out - the minimum value of all pixels in the cluster.
	* \param max_value			Out - the maximum value of all pixels in the cluster.
	* \param pixels_in_cluster	Out - the count of all pixels in the cluster.
	*/
	void getHsvRangeForCluster(
	         const cv::Mat &image,
	         const cv::Mat& labels,
	         int cluster_index,
	         uchar &min_hue,
	         uchar &max_hue,
	         uchar &min_saturation,
	         uchar &max_saturation,
	         uchar &min_value,
	         uchar &max_value,
	         int &pixels_in_cluster);

	cv::Mat labels_; // Maps each pixel in image to the result cluster index.
	image_transport::ImageTransport it_;  // OpenCV image transport.

	// Since the image is processed ansynchronously on another thread from the service handler,
	// This flag is used to indicate when the image processing is complete.
	bool image_processed_;

	// An indication if the ansynchronous image processing succeeded.
	RESULT_T result_;

	// An ok/error message from the asynchronous image processing.
	std::string result_msg_;

	/*! \brief Handle the service request to compute kmeans for the next video stream frame.
	* \param request 	The service request for the ROS service framework. See also ComputeKmeans.srv.
	* \param reponse 	The service response for the ROS service framework. See also ComputeKmeans.srv.
	* Return true iff the kmeans computation succeeded.
	*/
	bool computeKmeansCb(victoria_perception::ComputeKmeans::Request &request,
	                     victoria_perception::ComputeKmeans::Response &response);

	/*! \brief A ROS topic framework message handler for the video stream.
	* \param msg 	One frame of the video stream.
	*/
	void imageTopicCb(const sensor_msgs::ImageConstPtr& msg);

	/*! \brief perform the OpenCV kmeans analysis on an image.
	* \param original_image 	The image to be analized.
	* Return true iff the computation was successful.
	*/
	RESULT_T kmeansImage(const cv::Mat &original_image);

	// Cannot copy or clone the class instance.
	KmeansService(KmeansService const&) : it_(nh_) {};
	KmeansService& operator=(KmeansService const&) {};

public:
	/*! \brief The class constructor. 
	*/
	KmeansService();	
};

#endif // __VICTORIA_PERCEPTION_KMEANS