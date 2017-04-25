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

#include <sensor_msgs/image_encodings.h>
#include "kmeans_service/kmeans_service.h"

KmeansService::KmeansService() :
    image_processed_(false),
    it_(nh_),
    resize_width_(320),
    result_(ERROR),
    result_msg_("NO MESSAGE"),
    result_set_("NO RESULT") {
    nh_ = ros::NodeHandle("~");

    assert(compute_kmeans_service_ = nh_.advertiseService("compute_kmeans", &KmeansService::computeKmeansCb, this));
}

// Perform the compute KMEANS service.
bool KmeansService::computeKmeansCb(victoria_perception::ComputeKmeans::Request &request,
                                    victoria_perception::ComputeKmeans::Response &response) {
    number_clusters_ = request.number_clusters;
    resize_width_ = request.resize_width;
    show_annotated_window_ = request.show_annotated_window;
    image_transport::Subscriber image_sub_;
    image_processed_ = false;

    // Setup a handler for the video stream.
    // When one image is processed, tear down the handler.
    assert(image_sub_ = it_.subscribe(request.image_topic_name, 1, &KmeansService::imageTopicCb, this));

    while (ros::ok() && !image_processed_) {
        // Wait until a frame is received and processed.
        ros::spinOnce();
    }

    response.result_msg = result_msg_;
    response.kmeans_result = result_set_;
    if (result_ == SUCCESS) {
        return true;
    } else {
        return false;
    }
}

// Process one image frame.
void KmeansService::imageTopicCb(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        
        // Process the image frame.
        if (cv_ptr->image.rows >= resize_width_) {
            result_ = kmeansImage(cv_ptr->image);
        } else {
            // Image not wide enough.
            result_msg_ = "Image is not wide enough to downsample";
            result_ = ERROR;
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("[KmeansService::imageTopicCb] cv_bridge exception: %s", e.what());
        result_msg_ = "Exception raised";
        result_ = ERROR;
    }

    image_processed_ = true;
}

void KmeansService::downSampleImage(const cv::Mat &original_image, cv::Mat& out_image) {
    // Downsample the image for a faster computation.
    int orig_width = original_image.cols;
    int orig_height = original_image.rows;
    int resampled_height = (int) ((orig_height * 1.0 * resize_width_) / orig_width);
    cv::Size resize_dimensions(resize_width_, resampled_height);   // Down sample size for faster computation.
    cv::Mat image;
    cv::resize(original_image, out_image, resize_dimensions);
}

void KmeansService::createAnnotatedImage(const cv::Mat &image, std::vector<cv::Vec3b> clusters) {
    int width = image.cols;
    cv::Mat annotated_image;    // For debugging purposes.

    image.copyTo(annotated_image); // For debugging purposes

    // Annotate the image with color boxes for each cluster.
    static const int box_height = image.rows / (number_clusters_ + 1);

    // False color the image based upon pixel cluster.
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            annotated_image.at<cv::Vec3b>(row, col) = clusters[labels_.at<int>((row * image.cols) + col)];
        }
    }

    // Place colored cluster key as rectangles along right side of annotated picture.
    for (int i = 0; i < number_clusters_; i++) {
        cv::rectangle(annotated_image, cv::Point(width - 25, i * box_height), cv::Point(width, (i * box_height) + (box_height - 1)), cv::Scalar(clusters[i]), CV_FILLED);
    }

    cv::namedWindow("KMEANS_annotated", cv::WINDOW_NORMAL);
    cv::imshow("KMEANS_annotated", annotated_image);
    cv::waitKey(20);
}

void KmeansService::createResultSet(const cv::Mat& image) {
    std::stringstream ss;

    for (int i = 0; i < number_clusters_; i++) {
        uchar min_hue = 0;
        uchar max_hue = 0;
        uchar min_saturation = 0;
        uchar max_saturation = 0;
        uchar min_value = 0;
        uchar max_value = 0;
        int pixels_in_cluster = 0;
        getHsvRangeForCluster(image,
                              labels_,
                              i,
                              min_hue,
                              max_hue,
                              min_saturation,
                              max_saturation,
                              min_value,
                              max_value,
                              pixels_in_cluster);
        ss << "{cluster=" << i;
        ss << ";min_hue=" << (int) min_hue;
        ss << ";max_hue=" << (int) max_hue;
        ss << ";min_saturation=" << (int) min_saturation;
        ss << ";max_saturation=" << (int) max_saturation;
        ss << ";min_value=" << (int) min_value;
        ss << ";max_value=" << (int) max_value;
        ss << ";pixels=" << pixels_in_cluster << "}";

        ROS_INFO("[ConeDetector::kmeansImage] cluster: %3d, min_h: %3d, max_h: %3d, min_s: %3d, max_s: %3d, min_v: %3d, max_v: %3d, pixels: %6d",
                 i, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value, pixels_in_cluster);
    }

    result_set_ = ss.str();
}

// Compute KMEANS on a frame.
KmeansService::RESULT_T KmeansService::kmeansImage(const cv::Mat &original_image) {
    cv::Mat image;
    downSampleImage(original_image, image);

    cv::Mat centers;
    cv::TermCriteria criteria {
        cv::TermCriteria::COUNT,    // type
        100,                        // maxCount
        1                           // epsilon
    };
    int height = image.rows;
    int width = image.cols;

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

    createResultSet(image);
    if (show_annotated_window_) {
        createAnnotatedImage(image, centers_u8c3);
    }

    result_msg_ = "OK";
    return SUCCESS;
}

KmeansService::ChannelRange KmeansService::getClusterStatistics(const cv::Mat &image, int cluster_index, int channel_index) {
    int width = image.cols;
    cv::MatConstIterator_<int> label_first = labels_.begin<int>();
    cv::MatConstIterator_<int> label_last = labels_.end<int>();
    int histogram[256];
    ChannelRange result;

    for (int i = 0; i < 256; i++) histogram[i] = 0;

    int point_number = 0;
    int selected_point_count = 0;
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
    int max_value = 0;
    int i_at_max = 0;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > max_value) {
            max_value = histogram[i];
            i_at_max = i;
        }
    }

    // Find min range.
    result.min = i_at_max;
    for (int i = i_at_max; i >= 0; i--) {
        if (histogram[i] < (max_value / 10)) break;
        result.min = i;
    }

    // Find max range.
    result.max = i_at_max;
    for (int i = i_at_max; i < 256; i++) {
        if (histogram[i] < (max_value / 10)) break;
        result.max = i;
    }

    result.count = selected_point_count;

    //### Debug code. TO BE REMOVED.
    if (channel_index == 0) {
        ROS_INFO("cluster: %d, channel: %d, min: %d, max: %d, selected_point_count: %d",
        cluster_index, channel_index, result.min, result.max, selected_point_count);
        std::ostringstream ss;
        for (int i = 0; i < 256; i++) {ss << histogram[i] << ",";};ss<<std::endl;
        ROS_INFO("histogram: %s", ss.str().c_str());
    }

    return result;
}

void KmeansService::getHsvRangeForCluster(
         const cv::Mat &image,
         const cv::Mat& labels,
         int cluster_index,
         uchar &min_hue,
         uchar &max_hue,
         uchar &min_saturation,
         uchar &max_saturation,
         uchar &min_value,
         uchar &max_value,
         int &pixels_in_cluster) {
    int width = image.cols;

    // Find the HSV range for all points in that cluster.
    cv::MatConstIterator_<int> label_first = labels.begin<int>();
    cv::MatConstIterator_<int> label_last = labels.end<int>();
    min_hue = 179;
    max_hue = 0;
    min_saturation = 255;
    max_saturation = 0;
    min_value = 255;
    max_value = 0;
    ChannelRange ChannelRange;
    ChannelRange = getClusterStatistics(image, cluster_index, 0);
    min_hue = ChannelRange.min;
    max_hue = ChannelRange.max;
    ChannelRange = getClusterStatistics(image, cluster_index, 1);
    min_saturation = ChannelRange.min;
    max_saturation = ChannelRange.max;
    ChannelRange = getClusterStatistics(image, cluster_index, 2);
    min_value = ChannelRange.min;
    max_value = ChannelRange.max;
    pixels_in_cluster = ChannelRange.count;
}


