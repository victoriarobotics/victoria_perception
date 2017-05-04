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

void ConeDetector::configCb(const victoria_perception::ConeDetectorConfig &config, uint32_t level) {
    ROS_INFO("[ConeDetector::configCb] Reconfigure Request: alow_hue_: %d, ahigh_hue_: %d"
             ", alow_saturation_: %d, ahigh_saturation_: %d"
             ", alow_value_: %d, ahigh_value_: %d"
             ", blow_hue_: %d, bhigh_hue_: %d"
             ", blow_saturation_: %d, bhigh_saturation_: %d"
             ", blow_value_: %d, bhigh_value_: %d"
             ", min_cone_area_: %d, max_cone_area_: %d"
             ", max_aspect_ratio_: %7.4f"
             ", poly_epsilon_: %7.4f"
             ", erode_kernel_size_: %d"
             ", debug_: %s", 
             config.alow_hue_, config.ahigh_hue_, 
             config.alow_saturation_, config.ahigh_saturation_, 
             config.alow_value_, config.ahigh_value_ ,
             config.blow_hue_, config.bhigh_hue_, 
             config.blow_saturation_, config.bhigh_saturation_, 
             config.blow_value_, config.bhigh_value_ ,
             config.min_cone_area_, config.max_cone_area_,
             config.max_aspect_ratio_,
             config.poly_epsilon_,
             config.erode_kernel_size_,
             config.debug_ ? "TRUE" : "FALSE");
    alow_hue_range_ = config.alow_hue_;
    ahigh_hue_range_ = config.ahigh_hue_;
    alow_saturation_range_ = config.alow_saturation_;
    ahigh_saturation_range_ = config.ahigh_saturation_;
    alow_value_range_ = config.alow_value_;
    ahigh_value_range_ = config.ahigh_value_;
    blow_hue_range_ = config.blow_hue_;
    bhigh_hue_range_ = config.bhigh_hue_;
    bhigh_saturation_range_ = config.bhigh_saturation_;
    blow_value_range_ = config.blow_value_;
    bhigh_value_range_ = config.bhigh_value_;
    blow_saturation_range_ = config.blow_saturation_;
    low_contour_area_ = config.min_cone_area_;
    high_contour_area_ = config.max_cone_area_;
    max_aspect_ratio_ = config.max_aspect_ratio_;
    poly_epsilon_ = config.poly_epsilon_;
    erode_kernel_size_ = config.erode_kernel_size_;
    debug_ = config.debug_;
}

bool ConeDetector::strToBgr(const std::string &bgr_string, cv::Scalar &out_color) {
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

bool ConeDetector::hullIsValid(const std::vector<cv::Point> &hull) {
    // Find the bounding rectangle so we can easily determine hull points which
    // are above vs. below the horizontal centerline.
    cv::Rect bounding_rect = cv::boundingRect(cv::Mat(hull));

    float y_center = bounding_rect.y + (bounding_rect.height / 2.0); // The height of the horizontal centerline.

    // Computer the upper/lower left/right x-coordinates of the hull. Like a bounding box only it's a 
    // tight fit to the hull
    int top_left_x = std::numeric_limits<int>::max();
    int top_right_x = std::numeric_limits<int>::min();
    int bottom_left_x = std::numeric_limits<int>::max();
    int bottom_right_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();  // And while we're looking at point hull points, compute the top and bottom y-positions in the hull.
    int max_y = std::numeric_limits<int>::min();
    for (const cv::Point &point : hull) {
        // Compute the topmost and bottommost y-coordinates in the hull.
        if (point.y < min_y) min_y = point.y;
        if (point.y > max_y) max_y = point.y;

        // Computer the upper/lower left/right x-coordinates in the hull.
        if (point.y < y_center) {
            if (point.x < top_left_x) top_left_x = point.x;
            if (point.x > top_right_x) top_right_x = point.x;
        } else {
            if (point.x < bottom_left_x) bottom_left_x = point.x;
            if (point.x > bottom_right_x) bottom_right_x = point.x;
        }
    }

    // We're looking for a trapezoid, so compute the length of the top width
    // and bottom width using the upper/lower left/right x-coordinates from above.
    int length_top = top_right_x - top_left_x;
    int length_bottom = bottom_right_x - bottom_left_x;

    // We will also want to compute the aspect ratio. For the height
    // from the topmost and bottommost points from above and for the
    // width we use the average of the length_top and length_bottom from above.
    int height = max_y - min_y;
    if (height == 0) height = 1;
    float aspect_ratio = ((length_top + length_bottom) / 2.0) / height; 

    std::stringstream ss;
    ss << "[ConeDetector::hullIsValid]";

    ROS_INFO_COND_NAMED(debug_, "cone_detector", " tl: %d, tr: %d, bl: %d, br: %d, miny: %d, maxy: %d, ltop: %d, lbot: %d, h: %d, ar: %5.3f",
        top_left_x, top_right_x, bottom_left_x, bottom_right_x, min_y, max_y, length_top, length_bottom, height, aspect_ratio);
    if (aspect_ratio > max_aspect_ratio_) {
        ss << " REJECT, aspect_ratio: " << std::setprecision(4) << aspect_ratio << " greater than max: " << std::setprecision(4) << max_aspect_ratio_;
        ROS_INFO_COND_NAMED(debug_, "cone_detector", "] %s", ss.str().c_str());
        return false;
    }

    bool result = (length_bottom > 0) &&
                  (length_top < (0.75 * length_bottom));
    if (result) {
        ss << " ACCEPTED";
    } else {
        if (length_bottom <= 0) ss << "REJECT zero bottom length. ";
        if (length_top >= (0.75 * length_bottom)) ss << "REJECT top longer that 0.75 of bottom. ";
    }
    
    ROS_INFO_COND_NAMED(debug_, "cone_detector", " %s", ss.str().c_str());
    return result;
}

void ConeDetector::imageCb(const cv::Mat &original_image) {
    static const cv::Scalar color = cv::Scalar(0, 0, 255);  // Color of annotated primary circle.
    static const cv::Scalar non_primary_color = cv::Scalar(0, 255, 255);    // Color of annotated non-primary circles.

    cv::Mat img_hsv;    // We'll work in HSV space for color detction.
    cv::Size resize_dimensions(resize_x_, resize_y_);   // Down sample size for faster computation.
    cv::Mat image;
    cv::Mat annotation_image;

    // Downsample the image for a faster computation.
    cv::resize(original_image, image, resize_dimensions);

    // Convert image to HSV space for easier color detection.
    cv::cvtColor(image, img_hsv, CV_BGR2HSV); // Convert the captured frame from BGR to HSV


    // Using color detection, convert interesting areas in picture to white and uninteresting areas to black.
    cv::Mat a_img_thresholded;
    cv::inRange(img_hsv, 
                cv::Scalar(alow_hue_range_, alow_saturation_range_, alow_value_range_), 
                cv::Scalar(ahigh_hue_range_, ahigh_saturation_range_, ahigh_value_range_), 
                a_img_thresholded);
    cv::Mat b_img_thresholded;
    cv::inRange(img_hsv, 
                cv::Scalar(blow_hue_range_, blow_saturation_range_, blow_value_range_), 
                cv::Scalar(bhigh_hue_range_, bhigh_saturation_range_, bhigh_value_range_), 
                b_img_thresholded);

    // Merge the two filters
    cv::Mat merged_img_thresholded;
    cv::Mat merged_img_thresholded_tmp;
    cv::bitwise_or(a_img_thresholded, b_img_thresholded, merged_img_thresholded_tmp);
    merged_img_thresholded_tmp.copyTo(merged_img_thresholded);

    // Remove small objects from the background.
    cv::erode(merged_img_thresholded, merged_img_thresholded, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_kernel_size_, erode_kernel_size_)));
    cv::dilate(merged_img_thresholded, merged_img_thresholded, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_kernel_size_, erode_kernel_size_)));
    cv::GaussianBlur(merged_img_thresholded, merged_img_thresholded, cv::Size(5, 5), 0);

    // Now convert white areas to a white outline.
    cv::Canny(merged_img_thresholded, merged_img_thresholded, 50, 150, 3);
    if (debug_){
        cv::namedWindow("A HSV", cv::WINDOW_AUTOSIZE);
        cv::imshow("A HSV", a_img_thresholded);
        cv::namedWindow("B HSV", cv::WINDOW_AUTOSIZE);
        cv::imshow("B HSV", b_img_thresholded);
        cv::namedWindow("MERGED_HSV", cv::WINDOW_AUTOSIZE);
        cv::imshow("MERGED_HSV", merged_img_thresholded_tmp);
        cv::namedWindow("SMEARED", cv::WINDOW_AUTOSIZE);
        cv::imshow("SMEARED", merged_img_thresholded);
    }

    // Find polygons that are closed loops of white lines.
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    double contourSize;
    cv::Mat tempImage;

    merged_img_thresholded.copyTo(tempImage);
    cv::findContours(tempImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> *polynomial_of_best_cone(NULL);

    // Now find the best cone (if any) in the image.
    object_detected_ = false;
    bool at_least_on_good_area_found = false;
    if (!contours.empty()) {
        size_t best_blob_index = -1;
        int best_blob_size = 0;

        for (size_t i = 0; i < contours.size(); i++) {
            std::vector<cv::Point> dp;

            // Simplify the contours into less-jagged polynomials.
            cv::approxPolyDP(cv::Mat(contours[i]), dp, poly_epsilon_, true);

            // Filter out polynomials that are too small or too large to be an interesting cone.
            int contour_area = cv::contourArea(contours[i]);
            if ((contour_area >= low_contour_area_) && (contour_area <= high_contour_area_)) {
                at_least_on_good_area_found = true;
                // Convert the polynomial into a convex hull (i.e., ignore any jagged areas that
                // point inward).
                std::vector<cv::Point> hull;
                cv::convexHull(cv::Mat(dp), hull, false);

                // Filter out convex hulls that have too few points to be a triangle or too many
                // points (too complex a shape) to be an interesting cone-shaped object.
                if ((hull.size() >= 3) && (hull.size() <= 10)) {
                    bool ok = hullIsValid(hull);    // Do further test on hull to see if it's a likely good cone.
                    if (ok) {
                        // We've got a winner. Draw a line around the hull in the annotation image.
                        cv::polylines(image, dp/*hull*/, true, cv::Scalar(255, 0, 0), 1, 8, 0);

                        // Capture information about the winner.
                        best_blob_index = i;
                        best_blob_size = contour_area;
                        polynomial_of_best_cone = &contours[i];
                        break;
                    } else {
                        // Not a good enough hull to be a cone candidate. Ignore it.
                        ROS_INFO_COND_NAMED(debug_, "cone_detector", "[ConeDetector::imageCb] hullIsValid() returns false");
                    }
                } else {
                    // Hull has too few points to be a triangle or too many points (shape is too complex). Ignore it.
                    ROS_INFO_COND_NAMED(debug_, "cone_detector", "[ConeDetector::imageCb] hull size: %ld not in [3..10)",
                        hull.size());
                }
            } else {
                // Contour area is too small or too large to be an interesting cone candidate. Ignore it.
            }
        }

        if (!at_least_on_good_area_found) {
            ROS_INFO_COND_NAMED(debug_, "cone_detector", "[ConeDetector::imageCb] not one contour had an area in [%d..%d]",
                low_contour_area_, high_contour_area_);

        }

        // Upscale the annotation image so that text annotations will be readable.
        cv::resize(image, annotation_image, cv::Size(640, 480));
        if (best_blob_index != -1) {
            // A good cone candidate was selected. Draw a circle around it in the annotation window.
            cv::Point2f center;
            float radius;
            cv::approxPolyDP(cv::Mat(contours[best_blob_index]), *polynomial_of_best_cone, 3, true);
            cv::minEnclosingCircle((cv::Mat) *polynomial_of_best_cone, center, radius);

            int scale_factor_x = 640 / resize_x_;
            int scale_factor_y = 480 / resize_y_;
            cv::Point2f scaled_center(center);
            scaled_center.x *= scale_factor_x;
            scaled_center.y *= scale_factor_y;

            cv::circle(annotation_image, scaled_center, (int) radius * scale_factor_x, color, 4, 8, 0 );

            // Draw a different circle around the other contours that were detected but rejected.
            for (size_t i = 0; i < contours.size(); i++) {
                if (i == best_blob_index) continue; // Ignore the contour that was accepted.
                cv::approxPolyDP(cv::Mat(contours[i]), *polynomial_of_best_cone, 3, true );
                float radius;
                cv::Point2f obj_center;
                cv::minEnclosingCircle((cv::Mat) *polynomial_of_best_cone, obj_center, radius );
                obj_center.x *= scale_factor_x;
                obj_center.y *= scale_factor_y;
                cv::circle(annotation_image, obj_center, (int) radius * scale_factor_x, non_primary_color, 1, 8, 0);
            }

            // Capture cone info to be published.
            object_x_ = center.x;
            object_y_ = center.y;
            image_width_ = image.cols;
            image_height_ = image.rows;
            object_detected_ = true;
            object_area_ = best_blob_size;

            // Annotate the upper left of the annotated image with the location and size of the detected cone.
            cv::Scalar info_color;
            bool info_color_ok = strToBgr("2222FF", info_color);
            std::stringstream info_msg;
            info_msg << "@" << object_x_ << "," << object_y_ << "=" << object_area_;
            cv::putText(annotation_image, info_msg.str(), cvPoint(4, 42), g_font_face, g_font_scale, info_color, g_font_line_thickness, 8, false);
            
            // Publish one version of the cone detector message.
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
            ROS_INFO_COND_NAMED(debug_, "cone_detector", "[ConeDetector::imageCb] Cone detected x: %d, y: %d, w: %d, h: %d, area: %d",
                object_x_, object_y_, image_width_, image_height_, object_area_);
        } else {
            ROS_INFO_COND_NAMED(debug_, "cone_detector", "[ConeDetector::imageCb] No cone detected");
            
            // Publish one version of the cone detector message.
            std::stringstream not_found_msg;
            not_found_msg << "ConeDetector:NotFound;X:0;Y:0;AREA:0;I:0;ROWS:"
                << image_height_
                << ";COLS:" << image_width_;
            std_msgs::String message;
            message.data = not_found_msg.str();
            cone_found_pub_.publish(message);
        }

        placeAnnotationsInImage(annotation_image);  // Put any user annotations in the annotated image.

        // Emit the annotated image.
        sensor_msgs::ImagePtr annotated_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", annotation_image).toImageMsg();
        image_pub_annotated_.publish(annotated_image_msg);
    } else {
        ROS_INFO_COND_NAMED(debug_, "cone_detector", "[ConeDetector] no contours found");
        cv::resize(image, annotation_image, cv::Size(640, 480));
        placeAnnotationsInImage(annotation_image);

        // Emit the annotated image.
        sensor_msgs::ImagePtr annotated_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", annotation_image).toImageMsg();
        image_pub_annotated_.publish(annotated_image_msg);

        // Publish one version of the cone detector message.
        std::stringstream not_found_msg;
        not_found_msg << "ConeDetector:NotFound;X:0;Y:0;AREA:0;I:0;ROWS:"
            << image_height_
            << ";COLS:" << image_width_;
        std_msgs::String message;
        message.data = not_found_msg.str();
        cone_found_pub_.publish(message);
    }

    // Emit the thresholded image.
    sensor_msgs::ImagePtr thresholded_image = cv_bridge::CvImage(std_msgs::Header(), "mono8", merged_img_thresholded).toImageMsg();
    image_pub_thresholded_.publish(thresholded_image);
    cv::waitKey(10);
}

void ConeDetector::placeAnnotationsInImage(cv::Mat &annotation_image) {
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
    alow_hue_range_(0),
    ahigh_hue_range_(0),
    alow_saturation_range_(0),
    ahigh_saturation_range_(0),
    alow_value_range_(0),
    ahigh_value_range_(0),
    blow_hue_range_(0),
    bhigh_hue_range_(0),
    blow_saturation_range_(0),
    bhigh_saturation_range_(0),
    blow_value_range_(0),
    bhigh_value_range_(0),
    erode_kernel_size_(3),
    low_contour_area_(500),
    high_contour_area_(200000),
    ll_annotation_(""),
    ll_color_(255, 255, 255),
    lr_annotation_(""),
    lr_color_(255, 255, 255),
    max_aspect_ratio_(0.85),
    poly_epsilon_(8),
    resize_x_(320),
    resize_y_(240),
    show_step_times_(false),
    ul_annotation_(""),
    ul_color_(255, 255, 255),
    ur_annotation_(""),
    ur_color_(255, 255, 255) {

    configCallbackType_ = boost::bind(&ConeDetector::configCb, this, _1, _2);
    dynamic_server_.setCallback(configCallbackType_);

    assert(ros::param::get("~camera_name", camera_name_));
    assert(ros::param::get("~debug_cone_detector", debug_));
    assert(ros::param::get("~erode_kernel_size_", erode_kernel_size_));
    assert(ros::param::get("~image_topic_name", image_topic_name_));
    assert(ros::param::get("~max_aspect_ratio", max_aspect_ratio_));
    assert(ros::param::get("~poly_epsilon", poly_epsilon_));
    assert(ros::param::get("~show_step_times", show_step_times_));

    assert(ros::param::get("~alow_hue_range", alow_hue_range_));
    assert(ros::param::get("~ahigh_hue_range", ahigh_hue_range_));
    assert(ros::param::get("~alow_saturation_range", alow_saturation_range_));
    assert(ros::param::get("~ahigh_saturation_range", ahigh_saturation_range_));
    assert(ros::param::get("~alow_value_range", alow_value_range_));
    assert(ros::param::get("~ahigh_value_range", ahigh_value_range_));
    assert(ros::param::get("~blow_hue_range", blow_hue_range_));
    assert(ros::param::get("~bhigh_hue_range", bhigh_hue_range_));
    assert(ros::param::get("~blow_saturation_range", blow_saturation_range_));
    assert(ros::param::get("~bhigh_saturation_range", bhigh_saturation_range_));
    assert(ros::param::get("~blow_value_range", blow_value_range_));
    assert(ros::param::get("~bhigh_value_range", bhigh_value_range_));
    assert(ros::param::get("~low_contour_area", low_contour_area_));
    assert(ros::param::get("~high_contour_area", high_contour_area_));

    assert(ros::param::get("~resize_x", resize_x_));
    assert(ros::param::get("~resize_y", resize_y_));

    ROS_INFO("[ConeDetector] PARAM camera_name: %s", camera_name_.c_str());
    ROS_INFO("[ConeDetector] PARAM debug_cone_detector: %s", debug_ ? "TRUE" : "FALSE");
    ROS_INFO("[ConeDetector] PARAM erode_kernel_size: %d", erode_kernel_size_);
    ROS_INFO("[ConeDetector] PARAM image_topic_name: %s", image_topic_name_.c_str());
    ROS_INFO("[ConeDetector] PARAM low_contour_area: %d, high_contour_area: %d", low_contour_area_, high_contour_area_);
    ROS_INFO("[ConeDetector] PARAM alow_hue_range: %d, ahigh_hue_range: %d", alow_hue_range_, ahigh_hue_range_);
    ROS_INFO("[ConeDetector] PARAM alow_saturation_range: %d, ahigh_saturation_range: %d, ", alow_saturation_range_, ahigh_saturation_range_);
    ROS_INFO("[ConeDetector] PARAM alow_value_range: %d, ahigh_value_range: %d", alow_value_range_, ahigh_value_range_);
    ROS_INFO("[ConeDetector] PARAM blow_hue_range: %d, bhigh_hue_range: %d", blow_hue_range_, bhigh_hue_range_);
    ROS_INFO("[ConeDetector] PARAM blow_saturation_range: %d, bhigh_saturation_range: %d, ", blow_saturation_range_, bhigh_saturation_range_);
    ROS_INFO("[ConeDetector] PARAM ablow_value_range: %d, bhigh_value_range: %d", blow_value_range_, bhigh_value_range_);
    ROS_INFO("[ConeDetector] PARAM max_aspect_ratio: %7.4f", max_aspect_ratio_);
    ROS_INFO("[ConeDetector] PARAM poly_epsilon: %7.4f", poly_epsilon_);
    ROS_INFO("[ConeDetector] PARAM resize_x: %d, resize_y: %d", resize_x_, resize_y_);
    ROS_INFO("[ConeDetector] PARAM show_step_times: %s", show_step_times_ ? "TRUE" : "FALSE");

    nh_ = ros::NodeHandle("~");

    assert(annotate_service_ = nh_.advertiseService("annotate_detector_image", &ConeDetector::annotateCb, this));

    assert(image_pub_annotated_ = it_.advertise("cone_detector/annotated_image", 1));
    assert(image_pub_thresholded_ = it_.advertise("cone_detector/thresholded_image", 1));
    assert(image_sub_ = it_.subscribe(image_topic_name_, 1, &ConeDetector::imageTopicCb, this));
    assert(cone_found_pub_ = nh_.advertise<std_msgs::String>("cone_detector_summary", 2));

    victoria_perception::ConeDetectorConfig config;
    config.alow_hue_ = alow_hue_range_;
    config.ahigh_hue_ = ahigh_hue_range_;
    config.alow_saturation_ = alow_saturation_range_;
    config.ahigh_saturation_ = ahigh_saturation_range_;
    config.alow_value_ = alow_value_range_;
    config.ahigh_value_ = ahigh_value_range_;
    config.blow_hue_ = blow_hue_range_;
    config.bhigh_hue_ = bhigh_hue_range_;
    config.blow_saturation_ = blow_saturation_range_;
    config.bhigh_saturation_ = bhigh_saturation_range_;
    config.blow_value_ = blow_value_range_;
    config.bhigh_value_ = bhigh_value_range_;
    config.min_cone_area_ = low_contour_area_;
    config.max_cone_area_ = high_contour_area_;
    config.max_aspect_ratio_ = max_aspect_ratio_;
    config.poly_epsilon_ = poly_epsilon_;
    config.erode_kernel_size_ = erode_kernel_size_;
    dynamic_server_.updateConfig(config);

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
