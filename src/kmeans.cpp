#include <iostream>
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

#include <cstdlib>
#include <limits.h>
#include <sensor_msgs/image_encodings.h>
#include "kmeans/kmeans.h"
#include "victoria_perception/KmeansFeedback.h"

Kmeans::Kmeans() :
     compute_kmeans_action_server_(nh_,
                                  "compute_kmeans",
                                  boost::bind(&Kmeans::computeKmeansCb, this, _1), false),
    image_processed_(false),
    it_(nh_),
    resize_width_(320),
    result_(ERROR),
    result_msg_("NO MESSAGE"),
    result_set_("NO RESULT") {
    // Create the topic publisher for the annotated image.
    assert(image_pub_annotated_ = it_.advertise("kmeans/annotated_image", 1, true /* latched */));

    // Begin the action server.
    compute_kmeans_action_server_.start();
}

// Perform the compute KMEANS service.
bool Kmeans::computeKmeansCb(const victoria_perception::KmeansGoalConstPtr &goal) {
    victoria_perception::KmeansResult result;
    attempts_ = goal->attempts;
    number_clusters_ = goal->number_clusters;
    resize_width_ = goal->resize_width;
    image_processed_ = false;

    victoria_perception::KmeansFeedback feedback;

    feedback.step = "{\"progress\": \"Setting up video stream handler, await frame processing\"}";
    compute_kmeans_action_server_.publishFeedback(feedback);

    // Setup a handler for the video stream.
    // When one image is processed, tear down the handler in imageTopicCb--done there to try to
    // ensure that only the desired number of frames is processed.
    assert(image_sub_ = it_.subscribe(goal->image_topic_name, 1, &Kmeans::imageTopicCb, this));

    while (ros::ok() && !image_processed_) {
        // Wait until a frame is received and processed.
        if (compute_kmeans_action_server_.isPreemptRequested()) {
            compute_kmeans_action_server_.setPreempted();
            return false;
        }

        ros::spinOnce();
    }

    result.result_msg = result_msg_;
    result.kmeans_result = result_set_;
    if (result_ == SUCCESS) {
        compute_kmeans_action_server_.setSucceeded(result);
        return true;
    } else {
        compute_kmeans_action_server_.setAborted();
        return false;
    }
}

// Process one image frame.
void Kmeans::imageTopicCb(const sensor_msgs::ImageConstPtr& msg) {
    victoria_perception::KmeansFeedback feedback;
    try {
        feedback.step = "{\"progress\": \"Processing one frame\"}";
        compute_kmeans_action_server_.publishFeedback(feedback);
        
        image_sub_.shutdown(); // Frame is received, tear down frame listener.

        // Convert to BGR format from RGB.
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        
        // Process the image frame.
        if (cv_ptr->image.rows >= resize_width_) {
            result_ = kmeansImage(cv_ptr->image);
        } else {
            // Image not wide enough.
            feedback.step = "{\"error\": \"Image is not wide enough to support downsample.\"}";
            compute_kmeans_action_server_.publishFeedback(feedback);
            compute_kmeans_action_server_.setAborted();
            result_msg_ = "Image is not wide enough to downsample";
            result_ = ERROR;
        }
    } catch (cv_bridge::Exception& e) {
        std::ostringstream ss;
        ss << "{\"exception: \": \"" << e.what() << "\"}";
        feedback.step = ss.str();
        compute_kmeans_action_server_.publishFeedback(feedback);
        compute_kmeans_action_server_.setAborted();
        result_msg_ = "Exception raised: " + *e.what();
        result_ = ERROR;
    }

    image_processed_ = true;
}

// Compute KMEANS on a frame.
Kmeans::RESULT_T Kmeans::kmeansImage(const cv::Mat &original_image) {
    cv::Mat image;
    downSampleImage(original_image, image);

    cv::Mat centers;
    cv::TermCriteria criteria {
        cv::TermCriteria::COUNT,    // type
        100,                        // maxCount
        1                           // epsilon
    };
    unsigned int height = image.rows;
    unsigned int width = image.cols;

    // Convert data into shape required for KMEANS.
    cv::Mat reshaped_image = image.reshape(1, image.cols * image.rows);
    assert(reshaped_image.type() == CV_8UC1);
    cv::Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);
    assert(reshaped_image32f.type() == CV_32FC1);

    // Do KMEANS analysis.
    cv::kmeans(reshaped_image32f,           // Floating-point matrix of input samples, one row per sample.
               number_clusters_,            // Number of clusters to split the set by.
               labels_,                     // Input/output integer array that stores the cluster indices for every sample.
               criteria,                    // The algorithm termination criteria.
               attempts_,                   // Number of times the algorithm is executed using different initial labellings.
               cv::KMEANS_RANDOM_CENTERS,   // Select random initial centers in each attempt.
               centers);                    // Output matrix of the cluster centers, one row per each cluster center.
    assert(labels_.type() == CV_32SC1);
    assert(centers.type() == CV_32FC1);

    // Convert centers to colors.
    cv::Mat centers_u8;
    centers.convertTo(centers_u8, CV_8UC1, 255.0); // Convert the original centers to an array of clusters, scaled from [0..1) => [0..255].
    cv::Mat centers_u8c3 = centers_u8.reshape(3);   // Convert to 3 columns (R, G, B).

    createResultSet(centers_u8c3, image);
    createAnnotatedImage(image, centers_u8c3);

    result_msg_ = "OK";
    return SUCCESS;
}

void Kmeans::downSampleImage(const cv::Mat &original_image, cv::Mat& out_image) {
    // Downsample the image for a faster computation.
    unsigned int orig_width = original_image.cols;
    unsigned int orig_height = original_image.rows;
    unsigned int resampled_height = (unsigned int) ((orig_height * 1.0 * resize_width_) / orig_width);
    cv::Size resize_dimensions(resize_width_, resampled_height);   // Down sample size for faster computation.
    cv::Mat image;
    cv::resize(original_image, out_image, resize_dimensions);
}

void Kmeans::createResultSet(const std::vector<cv::Vec3b>& clusters, const cv::Mat& image) {
    std::stringstream ss;
    ss << "[";

    for (unsigned int i = 0; i < number_clusters_; i++) {
        cv::Vec3b  point_bgr = clusters[i];

        uchar min_hue = 0;
        uchar max_hue = 0;
        uchar min_saturation = 0;
        uchar max_saturation = 0;
        uchar min_value = 0;
        uchar max_value = 0;
        unsigned int pixels_in_cluster = 0;
        getHsvRangeForCluster(clusters,
                              image,
                              labels_,
                              i,
                              min_hue,
                              max_hue,
                              min_saturation,
                              max_saturation,
                              min_value,
                              max_value,
                              pixels_in_cluster);
        ss << "{\"cluster\":" << i;
        ss << ",\"min_hue\":" << (unsigned int) min_hue;
        ss << ",\"max_hue\":" << (unsigned int) max_hue;
        ss << ",\"min_saturation\":" << (unsigned int) min_saturation;
        ss << ",\"max_saturation\":" << (unsigned int) max_saturation;
        ss << ",\"min_value\":" << (unsigned int) min_value;
        ss << ",\"max_value\":" << (unsigned int) max_value;
        ss << ",\"pixels\":" << pixels_in_cluster;
        ss << ",\"color_rgb\":[" << int(point_bgr[2]) << "," << int(point_bgr[1]) << "," << int(point_bgr[0]) << "]";
        ss  << "}";
        if (i < (number_clusters_ - 1)) ss << ",";
    }

    ss << "]";
    result_set_ = ss.str();
}

void Kmeans::getHsvRangeForCluster(
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
     unsigned int &pixels_in_cluster) {
    unsigned int width = image.cols;

    // Find the HSV range for all points in that cluster.
    min_hue = 179;
    max_hue = 0;
    min_saturation = 255;
    max_saturation = 0;
    min_value = 255;
    max_value = 0;
    ChannelRange ChannelRange;
    ChannelRange = getClusterStatistics(clusters, image, cluster_index, 0);
    min_hue = ChannelRange.min;
    max_hue = ChannelRange.max;
    ChannelRange = getClusterStatistics(clusters, image, cluster_index, 1);
    min_saturation = ChannelRange.min;
    max_saturation = ChannelRange.max;
    ChannelRange = getClusterStatistics(clusters, image, cluster_index, 2);
    min_value = ChannelRange.min;
    max_value = ChannelRange.max;
    pixels_in_cluster = ChannelRange.count;
}

typedef struct HIST_DATA { unsigned int count; unsigned int value; HIST_DATA(int c, int x) : count(c), value(x){};} HIST_DATA;
bool sortHistData(const HIST_DATA& l, const HIST_DATA& r) { return l.count > r.count; }

Kmeans::ChannelRange Kmeans::getClusterStatistics(
        const std::vector<cv::Vec3b>& clusters,
        const cv::Mat &image, 
        unsigned int cluster_index, 
        unsigned int channel_index) {
    unsigned int width = image.cols;
    cv::MatConstIterator_<unsigned int> label_first = labels_.begin<unsigned int>();
    cv::MatConstIterator_<unsigned int> label_last = labels_.end<unsigned int>();
    unsigned int histogram[256];
    ChannelRange result;

    for (unsigned int i = 0; i < 256; i++) histogram[i] = 0;

    unsigned int point_number = 0;
    unsigned int selected_point_count = 0;
    while (label_first != label_last) {
        if (*label_first == cluster_index) {
            cv::Mat bgr_mat = image(cv::Rect(point_number % width, point_number / width, 1, 1));
            cv::Mat hsv_mat;
            cv::cvtColor(bgr_mat, hsv_mat, cv::COLOR_BGR2HSV);
            cv::Vec3b  point_hsv = hsv_mat.at<cv::Vec3b>(0, 0);
            histogram[point_hsv[channel_index]]++;
            selected_point_count++;
        }

        point_number++;
        label_first++;
    }

    // Find peak.
    std::vector<HIST_DATA> data_set;
    unsigned int max_value = 0;
    unsigned int i_at_max = 0;
    for (unsigned int i = 0; i < 256; i++) {
        data_set.push_back(HIST_DATA(histogram[i], i));
        if (histogram[i] > max_value) {
            max_value = histogram[i];
            i_at_max = i;
        }
    }

    std::sort(data_set.begin(), data_set.end(), sortHistData);

    unsigned int value_min = UINT_MAX;
    unsigned int value_max = 0;
    unsigned int count_sum = 0;
    unsigned int desired_sum = int(selected_point_count * 0.90);
    for (std::vector<HIST_DATA>::iterator it = data_set.begin(); it < data_set.end(); it++) {
        if (it->value < value_min) value_min = it->value;
        if (it->value > value_max) value_max = it->value;
        count_sum += it->count;
        // ROS_INFO("count: %d, value: %d, count_sum: %d, value_min: %d, value_max: %d",
        //          it->count, it->value, count_sum, value_min, value_max);//#####
        if (count_sum >= desired_sum) break;
    }

    result.min = value_min;
    result.max = value_max;

    // // Find min range.
    // result.min = i_at_max;
    // for (int i = i_at_max; i >= 0; i--) {
    //     if (histogram[i] < (max_value / 10)) break;
    //     result.min = i;
    // }

    // // Find max range.
    // result.max = i_at_max;
    // for (unsigned int i = i_at_max; i < 256; i++) {
    //     if (histogram[i] < (max_value / 10)) break;
    //     result.max = i;
    // }

    result.count = selected_point_count;

    victoria_perception::KmeansFeedback feedback;
    std::ostringstream ss;
    ss << "{";
    ss << "\"cluster\": " << cluster_index << ",";
    ss << "\"channel\": " << channel_index << ",";
    ss << "\"min\": " << result.min << ",";
    ss << "\"max\": " << result.max << ",";
    ss << "\"selected_point_count\": " << selected_point_count << ",";
    ss << "\"histogram\": [";
    for (unsigned int i = 0; i < 256; i++) {
        if (i > 0) ss << ",";
        ss << histogram[i];
    }

    ss << "]}";
    feedback.step = ss.str();
    compute_kmeans_action_server_.publishFeedback(feedback);

    return result;
}

void Kmeans::createAnnotatedImage(const cv::Mat &image, const std::vector<cv::Vec3b>& clusters) {
    unsigned int width = image.cols;
    cv::Mat annotated_image;    // For debugging purposes.

    image.copyTo(annotated_image); // For debugging purposes

    // Annotate the image with color boxes for each cluster.
    static const unsigned int box_height = image.rows / (number_clusters_ + 1);

    // False color the image based upon pixel cluster.
    for (unsigned int row = 0; row < image.rows; row++) {
        for (unsigned int col = 0; col < image.cols; col++) {
            annotated_image.at<cv::Vec3b>(row, col) = clusters[labels_.at<unsigned int>((row * image.cols) + col)];
        }
    }

    // Place colored cluster key as rectangles along right side of annotated picture.
    for (unsigned int i = 0; i < number_clusters_; i++) { 
        cv::rectangle(annotated_image, cv::Point(width - 25, i * box_height), cv::Point(width, (i * box_height) + (box_height - 1)), cv::Scalar(clusters[i]), CV_FILLED);
    }

    // Emit the annotated image.
    sensor_msgs::ImagePtr annotated_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", annotated_image).toImageMsg();
    image_pub_annotated_.publish(annotated_image_msg);
}


