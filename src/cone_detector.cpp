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

#include <cassert>
#include <opencv2/core/core.hpp>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <sys/stat.h>
#include <cstdlib>

#include <camera_info_manager/camera_info_manager.h>

#include "cone_detector/cone_detector.h"
#include <boost/algorithm/string.hpp>
#include <vector>
#include "victoria_perception/ObjectDetector.h"

void ConeDetector::configCb(victoria_perception::ConeDetectorConfig &config, uint32_t level) {
    ROS_INFO("Reconfigure Request: low_hue_: %d, high_hue_: %d"
             ", low_saturation_: %d, high_saturation_: %d"
             ", low_value_: %d, high_value_: %d"
             ", min_cone_area_: %d, max_cone_area_: %d"
             ", max_aspect_ratio_: %7.4f", 
             config.low_hue_, config.high_hue_, 
             config.low_saturation_, config.high_saturation_, 
             config.low_value_, config.high_value_ ,
             config.min_cone_area_, config.max_cone_area_,
             config.max_aspect_ratio_);
    low_hue_range_ = config.low_hue_;
    high_hue_range_ = config.high_hue_;
    low_saturation_range_ = config.low_saturation_;
    high_saturation_range_ = config.high_saturation_;
    low_value_range_ = config.low_value_;
    high_value_range_ = config.high_value_;
    low_contour_area_ = config.min_cone_area_;
    high_contour_area_ = config.max_cone_area_;
    max_aspect_ratio_ = config.max_aspect_ratio_;
}

bool ConeDetector::strToBgr(std::string bgr_string, cv::Scalar& out_color) {
    char* parse_end;
    long bgr = strtoll(bgr_string.c_str(), &parse_end, 16);
    out_color = cv::Scalar((bgr >> 16) & 0xFF, (bgr >> 8) & 0xFF, bgr & 0xFF);
    return true;
}

bool ConeDetector::annotateCb(victoria_perception::AnnotateDetectorImage::Request &request,
                              victoria_perception::AnnotateDetectorImage::Response &response) {
    std::vector<std::string> fields;
    boost::split(fields, request.annotation, boost::is_any_of(";"));
    if (fields.size() != 3) {
        response.result = "Invalid request format. Expected three semicolon-separated fields";
        return false;
    }

    if (boost::iequals(fields[0], "LL")) {
        ll_annotation_ = fields[2];
        bool color_ok = strToBgr(fields[1], ll_color_);
        if (!color_ok) {
            response.result = "Invalid request, seconf field is not a valid 6-character hex BGR value";
            return false;
        }
    } else if (boost::iequals(fields[0], "LR")) {
        lr_annotation_ = fields[2];
        bool color_ok = strToBgr(fields[1], lr_color_);
        if (!color_ok) {
            response.result = "Invalid request, seconf field is not a valid 6-character hex BGR value";
            return false;
        }
    } else if (boost::iequals(fields[0], "UL")) {
        ul_annotation_ = fields[2];
        bool color_ok = strToBgr(fields[1], ul_color_);
        if (!color_ok) {
            response.result = "Invalid request, seconf field is not a valid 6-character hex BGR value";
            return false;
        }
    } else if (boost::iequals(fields[0], "UR")) {
        ur_annotation_ = fields[2];
        bool color_ok = strToBgr(fields[1], ur_color_);
        if (!color_ok) {
            response.result = "Invalid request, seconf field is not a valid 6-character hex BGR value";
            return false;
        }
    } else {
        response.result = "Invalid request, first field is not LL, LR, UL or UR";
        return false;
    }

    response.result = "OK";
    return true;
}

bool ConeDetector::hullIsValid(std::vector<cv::Point>& hull) {
    cv::Rect bounding_rect = cv::boundingRect(cv::Mat(hull));
    //float aspect_ratio = 1.0 * bounding_rect.width / bounding_rect.height;
    float y_center = bounding_rect.y + (bounding_rect.height / 2.0);
    int top_left_x = 10000;
    int top_right_x = -10000;
    int bottom_left_x = 10000;
    int bottom_right_x = -10000;
    int min_y = 10000;
    int max_y = -1000;
    for (cv::Point point : hull) {
        if (point.y < min_y) min_y = point.y;
        if (point.y > max_y) max_y = point.y;
        if (point.y < y_center) {
            if (point.x < top_left_x) top_left_x = point.x;
            if (point.x > top_right_x) top_right_x = point.x;
        } else {
            if (point.x < bottom_left_x) bottom_left_x = point.x;
            if (point.x > bottom_right_x) bottom_right_x = point.x;
        }
    }

    int length_top = top_right_x - top_left_x;
    int length_bottom = bottom_right_x - bottom_left_x;
    int height = max_y - min_y;
    if (height == 0) height = 1;
    float aspect_ratio = ((length_top + length_bottom) / 2.0) / height; 

    std::stringstream ss;
    ss << "[hullIsValid]  tlx: " << top_left_x << ", trx: " << top_right_x << ", blx: " << bottom_left_x << ", brx" << bottom_right_x;
    ss << ", tlen: " << length_top << ", blen: " << length_bottom;
    ss << ", min_y: " << min_y << ", max_y: " << max_y << ", height: " << height;
    ss << ", pt asp ratio: " << aspect_ratio;
    double tl_ratio = (length_top / (1.0 * (length_bottom != 0 ? length_bottom : 1)));
    ss << ", tl_ratio: " << std::setprecision(4) << tl_ratio;
    if (aspect_ratio > max_aspect_ratio_) {
        ss << ", REJECT, aspect_ration greater than: " << std::setprecision(4) << max_aspect_ratio_;
        ROS_INFO("%s", ss.str().c_str());
        return false;
    }

   if (length_top < 0) ss << ", REJECT bad tlen";
    if (length_bottom <= 0) ss << ", REJECT bad blen";
    if (length_top >= (0.75 * length_bottom)) ss << ", REJECT bad shape";
    ROS_INFO("%s", ss.str().c_str());
    bool result = (length_top >= 0) &&
                  (length_bottom > 0) &&
                  (length_top < (0.75 * length_bottom));
    return result;
}

void ConeDetector::imageCb(cv::Mat& original_image) {
    static const cv::Scalar color = cv::Scalar(0, 0, 255);
    static const cv::Scalar non_primary_color = cv::Scalar(0, 255, 255);

    clock_t start;
    double duration_resize = 0;         // Time taken for resize call.
    double duration_cvtColor = 0;       // Time taken for cvtColor call.
    double duration_erode_dilate_ = 0;  // Time taken for erode/dilate calls.
    double duration_inRange = 0;        // Time taken for inRange call.
    double duration_copyTo = 0;         // Time taken for copyTo operation.
    double d_theta = 0;    // Time taken to find largets blob.
    double duration_imshow = 0;         // Time taken for imshow calls
    double duration_findContours = 0;   // Time taken for findContours call.
    double duration_find_largest = 0;   // Time taken to find the largest contour.

    cv::Mat img_HSV;
    if (show_step_times_) start = clock();
    cv::Size resize_dimensions(resize_x_, resize_y_);
    cv::Mat image;
    cv::Mat annotation_image;
    if (show_step_times_) start = clock();
    cv::resize(original_image, image, resize_dimensions);
    if (show_step_times_) duration_resize = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    cv::cvtColor(image, img_HSV, CV_BGR2HSV); // Convert the captured frame from BGR to HSV
    if (show_step_times_) duration_cvtColor = (std::clock() - start) / (double)CLOCKS_PER_SEC;


    cv::Mat imgThresholded;
    if (show_step_times_) start = clock();
    cv::inRange(img_HSV, 
                cv::Scalar(low_hue_range_, low_saturation_range_, low_value_range_), 
                cv::Scalar(high_hue_range_, high_saturation_range_, high_value_range_), imgThresholded); //Threshold the image
    if (show_step_times_) duration_inRange = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    if (show_step_times_) start = clock();
    // Remove small objects from the background.
    cv::erode(imgThresholded, imgThresholded, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::dilate(imgThresholded, imgThresholded, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::GaussianBlur(imgThresholded, imgThresholded, cv::Size(5, 5), 0);
    cv::Canny(imgThresholded, imgThresholded, 50, 150, 3);
    if (show_step_times_) duration_erode_dilate_ = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    double contourSize;
    cv::Mat tempImage;

    if (show_step_times_) start = clock();
    imgThresholded.copyTo(tempImage);
    if (show_step_times_) duration_copyTo = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    if (show_step_times_) start = clock();
    cv::findContours(tempImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if (show_step_times_) duration_findContours = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    std::vector<cv::Point> *best_cone = NULL;

    if (show_step_times_) start = clock();
    object_detected_ = false;
    if (!contours.empty()) {
        // Find best blob.
        size_t best_blob_index = -1;
        int best_blob_size = 0;

        //###std::vector<std::vector<cv::Point> > cones;
        for (size_t i = 0; i < contours.size(); i++) {
            std::stringstream ss;
            ss << "[imageCb] [" << i << "@" << contours[i][0].x << "x" << contours[i][0].y << "] dp: ";
            std::vector<cv::Point> dp;
            approxPolyDP(cv::Mat(contours[i]), dp, poly_epsilon_, true);
            for (cv::Point p : dp) {
                ss << p.x << "x" << p.y << " ";
            }
            int contour_size = cv::contourArea(contours[i]);
            if ((contour_size >= low_contour_area_) && (contour_size <= high_contour_area_)) {
                ss << " Good contour area: " << contour_size;
                ss << ", at " << contours[i][0].x << "x" << contours[i][0].y << ", hull: ";
                std::vector<cv::Point> hull;
                cv::convexHull(cv::Mat(dp), hull, false);
                for (cv::Point p : hull) {
                    ss << p.x << "x" << p.y << " ";
                }
                if ((hull.size() >= 3) && (hull.size() <= 10)) {

                    bool ok = hullIsValid(hull);
                    if (ok) {
                        cv::polylines(image, hull, true, cv::Scalar(255, 0, 0), 1, 8, 0);
                        best_blob_index = i;
                        best_blob_size = contour_size;
                        best_cone = &contours[i];
                        ss << " GOOD blob_size: " << best_blob_size;
                        break;
                    } else {
                        ss << "REJECT invalid hull";
                    }
                } else {
                    ss << "REJECT hull size: " << hull.size() << " not in [3 .. 10]";
                }

                ROS_INFO("%s", ss.str().c_str());
            } else {
                //ss << "REJECT contour_size: " << contour_size << " not in [" << low_contour_area_ << ", " << high_contour_area_ << "]";
                // ROS_INFO("%s", ss.str().c_str());
            }
        }


        // TODO:
        // MoveToCone  should ignore transient cone detects.
        // Capture all cones and select best one.
        // +Go back to 640 scale.
        // +No annotation if no cone detected--fix this
        // +Remove divide by zero possibility.
        // +Empty JPEG image errors.


        cv::resize(image, annotation_image, cv::Size(640, 480));
        //image.copyTo(annotation_image);
        if (best_blob_index != -1) {
            cv::Point2f center;
            float radius;
            approxPolyDP(cv::Mat(contours[best_blob_index]), *best_cone, 3, true);
            minEnclosingCircle((cv::Mat) *best_cone, center, radius);

            object_x_ = center.x;
            object_y_ = center.y;

            image_width_ = image.cols;
            image_height_ = image.rows;
            object_detected_ = true;
            object_area_ = best_blob_size;

            int scale_factor_x = 640 / resize_x_;
            int scale_factor_y = 480 / resize_y_;
            cv::Point2f scaled_center(center);
            scaled_center.x *= scale_factor_x;
            scaled_center.y *= scale_factor_y;

            circle(annotation_image, scaled_center, (int) radius * scale_factor_x, color, 4, 8, 0 );
            ROS_INFO("best index: %d, center: %dx%d, radius: %7.4f", (int) best_blob_index, (int) center.x, (int)center.y, radius);
            for (size_t i = 0; i < contours.size(); i++) {
                if (i == best_blob_index) continue;
                cv::approxPolyDP(cv::Mat(contours[i]), *best_cone, 3, true );
                float radius;
                cv::Point2f obj_center;
                minEnclosingCircle((cv::Mat) *best_cone, obj_center, radius );
                obj_center.x *= scale_factor_x;
                obj_center.y *= scale_factor_y;
                cv::circle(annotation_image, obj_center, (int) radius * scale_factor_x, non_primary_color, 1, 8, 0);
            }

            cv::Scalar info_color;
            bool info_color_ok = strToBgr("2222FF", info_color);
            std::stringstream info_msg;
            info_msg << "@" << object_x_ << "," << object_y_ << "=" << object_area_;
            cv::putText(annotation_image, info_msg.str(), cvPoint(4, 42), g_font_face, g_font_scale, info_color, g_font_line_thickness, 8, false);
            std::stringstream topic_msg;
            topic_msg << "ConeDetector:Found;X:" << object_x_
                << ";Y:" << object_y_
                << ";AREA:" << best_blob_size
                << ";I:" << best_blob_index
                << ";ROWS:" << image_height_
                << ";COLS:" << image_width_;
            std_msgs::String message;
            message.data = topic_msg.str();
            cone_found_pub_.publish(message);
        } else {
            ROS_INFO("best_blob_index is -1");
            std::stringstream not_found_msg;
            not_found_msg << "ConeDetector:NotFound;X:0;Y:0;AREA:0;I:0;ROWS:"
                << image_height_
                << ";COLS:" << image_width_;
            std_msgs::String message;
            message.data = not_found_msg.str();
            cone_found_pub_.publish(message);
        }

        placeAnnotationsInImage(annotation_image);

        sensor_msgs::ImagePtr annotated_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", annotation_image).toImageMsg();
        image_pub_annotated_.publish(annotated_image_msg);

        if (show_step_times_) duration_find_largest = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    } else {
        ROS_INFO("[imageCb] no contours found");
        cv::resize(image, annotation_image, cv::Size(640, 480));
        placeAnnotationsInImage(annotation_image);
        sensor_msgs::ImagePtr annotated_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", annotation_image).toImageMsg();
        image_pub_annotated_.publish(annotated_image_msg);
        std::stringstream not_found_msg;
        not_found_msg << "ConeDetector:NotFound;X:0;Y:0;AREA:0;I:0;ROWS:"
            << image_height_
            << ";COLS:" << image_width_;
        std_msgs::String message;
        message.data = not_found_msg.str();
        cone_found_pub_.publish(message);
    }

    if (show_step_times_) start = clock();
    sensor_msgs::ImagePtr thresholded_image = cv_bridge::CvImage(std_msgs::Header(), "mono8", imgThresholded).toImageMsg();
    image_pub_thresholded_.publish(thresholded_image);

    duration_imshow = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    if (show_step_times_) ROS_INFO("durations resize: %7.5f, "
                                   "cvtColor: %7.5f"
                                   ", inRange: %7.5f"
                                   ", erode/dilate: %7.5f"
                                   ", findLargest: %7.5f"
                                   ", showWindows: %7.5f"
                                   ", copyTo: %7.5f"
                                   ", findContours: %7.5f",
                                   duration_resize,
                                   duration_cvtColor,
                                   duration_inRange,
                                   duration_erode_dilate_,
                                   duration_find_largest,
                                   duration_imshow,
                                   duration_copyTo,
                                   duration_findContours);
    cv::waitKey(10);
}

void ConeDetector::placeAnnotationsInImage(cv::Mat annotation_image) {
    if (ll_annotation_.length() > 0) {
        cv::putText(annotation_image, ll_annotation_, cvPoint(4, annotation_image.rows - 10), g_font_face, g_font_scale, ll_color_, g_font_line_thickness, 8, false);
    }

    if (lr_annotation_.length() > 0) {
        cv::putText(annotation_image, lr_annotation_, cvPoint(annotation_image.cols / 2 + 4, annotation_image.rows - 10), g_font_face, g_font_scale, lr_color_, g_font_line_thickness, 8, false);
    }

    if (ul_annotation_.length() > 0) {
        cv::putText(annotation_image, ul_annotation_, cvPoint(4, 20), g_font_face, g_font_scale, ul_color_, g_font_line_thickness, 8, false);
    }

    if (ur_annotation_.length() > 0) {
        cv::putText(annotation_image, ur_annotation_, cvPoint(annotation_image.cols / 2 + 4, 20), g_font_face, g_font_scale, ur_color_, g_font_line_thickness, 8, false);
    }
}

ConeDetector::ConeDetector() :
    it_(nh_),
    low_hue_range_(0),
    high_hue_range_(44),
    low_saturation_range_(181),
    high_saturation_range_(255),
    low_value_range_(169),
    high_value_range_(255),
    low_contour_area_(500),
    high_contour_area_(200000),
    show_step_times_(false),
    ll_annotation_(""),
    ll_color_(255, 255, 255),
    lr_annotation_(""),
    lr_color_(255, 255, 255),
    max_aspect_ratio_(0.85),
    poly_epsilon_(6.7),
    resize_x_(320),
    resize_y_(240),
    ul_annotation_(""),
    ul_color_(255, 255, 255),
    ur_annotation_(""),
    ur_color_(255, 255, 255) {

    configCallbackType_ = boost::bind(&ConeDetector::configCb, this, _1, _2);
    dynamic_server_.setCallback(configCallbackType_);

    assert(ros::param::get("~camera_name", camera_name_));
    assert(ros::param::get("~image_topic_name", image_topic_name_));
    assert(ros::param::get("~max_aspect_ratio", max_aspect_ratio_));
    assert(ros::param::get("~poly_epsilon", poly_epsilon_));
    assert(ros::param::get("~show_step_times", show_step_times_));

    assert(ros::param::get("~low_hue_range", low_hue_range_));
    assert(ros::param::get("~high_hue_range", high_hue_range_));
    assert(ros::param::get("~low_saturation_range", low_saturation_range_));
    assert(ros::param::get("~high_saturation_range", high_saturation_range_));
    assert(ros::param::get("~low_value_range", low_value_range_));
    assert(ros::param::get("~high_value_range", high_value_range_));
    assert(ros::param::get("~low_contour_area", low_contour_area_));
    assert(ros::param::get("~high_contour_area", high_contour_area_));

    assert(ros::param::get("~resize_x", resize_x_));
    assert(ros::param::get("~resize_y", resize_y_));

    ROS_INFO("[ConeDetector] PARAM camera_name: %s", camera_name_.c_str());
    ROS_INFO("[ConeDetector] PARAM image_topic_name: %s", image_topic_name_.c_str());
    ROS_INFO("[ConeDetector] PARAM low_contour_area: %d, high_contour_area: %d", low_contour_area_, high_contour_area_);
    ROS_INFO("[ConeDetector] PARAM low_hue_range: %d, high_hue_range: %d", low_hue_range_, high_hue_range_);
    ROS_INFO("[ConeDetector] PARAM low_saturation_range: %d, high_saturation_range: %d, ", low_saturation_range_, high_saturation_range_);
    ROS_INFO("[ConeDetector] PARAM low_value_range: %d, high_value_range: %d", low_value_range_, high_value_range_);
    ROS_INFO("[ConeDetector] PARAM max_aspect_ratio: %7.4f", max_aspect_ratio_);
    ROS_INFO("[ConeDetector] PARAM poly_epsilon: %7.4f", poly_epsilon_);
    ROS_INFO("[ConeDetector] PARAM resize_x: %d, resize_y: %d", resize_x_, resize_y_);
    ROS_INFO("[ConeDetector] PARAM show_step_times: %s", show_step_times_ ? "TRUE" : "FALSE");

    victoria_perception::ConeDetectorConfig config;
    config.low_hue_ = low_hue_range_;
    config.high_hue_ = high_hue_range_;
    config.low_saturation_ = low_saturation_range_;
    config.high_saturation_ = high_saturation_range_;
    config.low_value_ = low_value_range_;
    config.high_value_ = high_value_range_;
    config.min_cone_area_ = low_contour_area_;
    config.max_cone_area_ = high_contour_area_;
    config.max_aspect_ratio_ = max_aspect_ratio_;
    dynamic_server_.updateConfig(config);

     nh_ = ros::NodeHandle("~");

    annotateService = nh_.advertiseService("annotate_detector_image", &ConeDetector::annotateCb, this);
    image_pub_annotated_ = it_.advertise("cone_detector/annotated_image", 1);
    image_pub_thresholded_ = it_.advertise("cone_detector/thresholded_image", 1);
    image_sub_ = it_.subscribe(image_topic_name_, 1, &ConeDetector::imageTopicCb, this);
    cone_found_pub_ = nh_.advertise<std_msgs::String>("cone_detector_summary", 2);
}

void ConeDetector::imageTopicCb(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("[ConeDetector::imageTopicCb] cv_bridge exception: %s", e.what());
        return;
    }

    // Process the image frame.
    if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60) {
        imageCb(cv_ptr->image);
    }
}

ConeDetector& ConeDetector::singleton() {
    static ConeDetector singleton_;
    return singleton_;
}

const double ConeDetector::g_font_scale = 0.75;
