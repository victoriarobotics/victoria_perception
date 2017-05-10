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

#include <actionlib/server/simple_action_server.h>
#include <algorithm>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

#include "victoria_perception/KmeansAction.h"

//* A Kmeans class.
/**
* This is a ROS action handler for the compute_kmeans action. 
* The action goal takes as parameters:
* (see Kmeans.action)
* int16 	attempts 			The 'attempts' parameter to the cv::kmeans method.
* string 	image_topic_name 	The topic name of the video stream.
* int16 	number_clusters 	The 'K' parameter of cv::kmeans.
* int16 	resize_width 		The downsample width of the original image to work with. 
*								It must be <= original image width. The height will be computed proportionately.
*
* Progress is reported in one field:
* string	step				This receives cluster histogram information in JSON format.
*
* Final result has two fields:
* string 	result_msg 			"OK" for good result, else an error message.
* string 	kmeans_result		The kmeans results in JSON format if result_msg is "OK".
*
* Also, an "annotated_image" is written to the "kmeans/annotated_image" topic. This image will be a downsampled
* version of the original image (see resize_width) where each pixel in the original image is replace with the
* color of the correspnding cluster. This give a posterized result. Also, along the right edge of the
* annotated image, a series of colored boxes is written, partially obscuring the original image. The
* topmost box corresponds to the color for cluster 0, the box below corresponds to the color for cluster 1
* and so on. You can use the cluster results produced in kmeans_result to map back to pixels in the original
* image using this color key;
*
* Here is a snippet of a sample kmeans_result with extra carriage returns for readability:
* {cluster= 0;min_hue=53;max_hue=53;min_saturation=4;max_saturation=4;min_value=234;max_value=235;pixels=9725}
* {cluster= 1;min_hue=60;max_hue=60;min_saturation=5;max_saturation=33;min_value=196;max_value=218;pixels=6630}
*
* Note that there are no outer curly braces or brackets to enclose the list.
*
* The action constructs a listener for the next frame in the video stream for the topic, does the computation, then
* uses the normal destructors to remove the listener.
*/
class Kmeans {
protected:
	/*!
	* \brief Holds the value range (min, max) and pxiels in a cluster for one channel in
	* a cluster (viz., hue channel, saturation channel or value channel).
	*/
	typedef struct ChannelRange {
	    unsigned int min;      // Min channel value.
	    unsigned int max;      // Max channel value.
	    unsigned int count;    // Number of pixels in the cluster.
	} ChannelRange;

	typedef actionlib::SimpleActionServer<victoria_perception::KmeansAction> KmeansActionServer;

	// ROS node handles.
	ros::NodeHandle nh_;								// Main ROS node handle.
	image_transport::Publisher image_pub_annotated_;	// For publishing the annotated image.
	image_transport::Subscriber image_sub_;				// Listen to the video stream.

	/*! \brief Handle the action request to compute kmeans for the next video stream frame.
	* \param goal 	The action goal for the ROS actionlib framework. See also the file KMeans.action.
	* Return true iff the kmeans computation succeeded.
	*/
	bool computeKmeansCb(const victoria_perception::KmeansGoalConstPtr &goal);

	KmeansActionServer compute_kmeans_action_server_;	// Handle to accept action requests.

	/*!
	* Indicates success or failure of a function.
	*/
	enum RESULT_T {
		SUCCESS,
		ERROR
	};

	// Parameters
	unsigned int attempts_;			// cv::kmeans 'attempts' value.
	std::string image_topic_name_;	// Topic name containing video stream.
	unsigned int number_clusters_;	// Number of clusters to form (cv::kmeans 'K' value).
	unsigned int resize_width_;		// Downsample image to this width (height will be proportional).
	std::string result_set_;		// The string response to be returned on success.

	/*! \brief Publish the annotated downsampled image where each pixel
	* is replaced by the color associated with a cluster. On the right
	* of the image, a sequence of colored boxes will be drawn. The topmost
	* box corresponds to cluster 0 and is the color of all the false colored
	* pixels in the annotated image for cluster 0. The box below corresponds
	* to the color of pixels for cluster 1 and so forth.
	* \param image 		The downsampled image which is copied and annotated.
	* \param clusters 	The list of colors for each cluster.
	*/
	void createAnnotatedImage(const cv::Mat &image, const std::vector<cv::Vec3b>& clusters);

	/*! \brief Create the result string for the action request.
	* \param clusters The clusters.
	* \param image 		The downsampled image. Used to find the colors of all pixels in a cluster.
	*/
	void createResultSet(const std::vector<cv::Vec3b>& clusters, const cv::Mat& image);

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
	* \param clusters 		The clusters.
	* \param image 			The downsampled image.
	* \param cluster_index 	Which cluster to provide statistic for.
	* \param channel_index	Which channel to provide statistic for. 0=>Hue, 1=>Saturation, 2=>Value.
	* The result is the min, max channel values and pixel count for all pixels in the cluster, for the given channel.
	*/
	ChannelRange getClusterStatistics(
		const std::vector<cv::Vec3b>& clusters, 
		const cv::Mat &image, 
		unsigned int cluster_index, 
		unsigned int channel_index);
	
	/*! \brief Get the min/max of hue, saturation and value for all pixels in a cluster, as well as
	* the count of all pixels in that cluster. See also getClusterStatistics for a note on the reduced range returned.
	* \param clusters 			The clusters.
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
	         const std::vector<cv::Vec3b>& clusters,
	         const cv::Mat &image,
	         const cv::Mat& labels,
	         unsigned int cluster_index,
	         uchar &min_hue,
	         uchar &max_hue,
	         uchar &min_saturation,
	         uchar &max_saturation,
	         uchar &min_value,
	         uchar &max_value,
	         unsigned int &pixels_in_cluster);

	cv::Mat labels_; // Maps each pixel in image to the result cluster index.
	image_transport::ImageTransport it_;  // OpenCV image transport.

	// Since the image is processed ansynchronously on another thread from the action request handler,
	// This flag is used to indicate when the image processing is complete.
	bool image_processed_;

	// An indication if the ansynchronous image processing succeeded.
	RESULT_T result_;

	// An ok/error message from the asynchronous image processing.
	std::string result_msg_;

	/*! \brief A ROS topic message handler for the video stream.
	* \param msg 	One frame of the video stream.
	*/
	void imageTopicCb(const sensor_msgs::ImageConstPtr& msg);

	/*! \brief perform the OpenCV kmeans analysis on an image.
	* \param original_image 	The image to be analized.
	* Return true iff the computation was successful.
	*/
	RESULT_T kmeansImage(const cv::Mat &original_image);

	//Cannot copy or clone the class instance.
	Kmeans(Kmeans const&) :
	    compute_kmeans_action_server_(nh_,
                                  "compute_kmeans",
                                  boost::bind(&Kmeans::computeKmeansCb, this, _1), false),
	    it_(nh_)
		{};
	Kmeans& operator=(Kmeans const&) {};

public:
	/*! \brief The class constructor. 
	*/
	Kmeans();

};

#endif // __VICTORIA_PERCEPTION_KMEANS