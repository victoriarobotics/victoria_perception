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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <sys/stat.h>
#include <cstdlib>

#include <camera_info_manager/camera_info_manager.h>

#include "cone_detector/cone_detector.h"
#include <boost/algorithm/string.hpp>
#include <vector>
#include "victoria_perception/ObjectDetector.h"

using namespace boost;
using namespace cv;
using namespace std;

bool ConeDetector::strToBgr(string bgr_string, Scalar& out_color) {
    char* parse_end;
    long bgr = strtoll(bgr_string.c_str(), &parse_end, 16);
    out_color = Scalar((bgr >> 16) & 0xFF, (bgr >> 8) & 0xFF, bgr & 0xFF);
    return true;
}

bool ConeDetector::annotateCb(victoria_perception::AnnotateDetectorImage::Request &request,
                              victoria_perception::AnnotateDetectorImage::Response &response) {
    vector<string> fields;
    split(fields, request.annotation, is_any_of(";"));
    ROS_INFO("[ConeDetector::annotate] request: %s, fields: %ld", request.annotation.c_str(), fields.size());
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

void ConeDetector::imageCb(Mat& original_image) {
    clock_t start;
    double duration_resize = 0;         // Time taken for resize call.
    double duration_cvtColor = 0;       // Time taken for cvtColor call.
    double duration_erode_dilate_ = 0;  // Time taken for erode/dilate calls.
    double duration_inRange = 0;        // Time taken for inRange call.
    double duration_copyTo = 0;         // Time taken for copyTo operation.
    double duration_find_argest = 0;    // Time taken to find largets blob.
    double duration_imshow = 0;         // Time taken for imshow calls
    double duration_findContours = 0;   // Time taken for findContours call.
    double duration_contoursPoly = 0;   // Time taken for contoursPoly call.

    Mat img_HSV;
    if (show_step_times_) start = clock();
    Size resize_dimensions(resize_x_, resize_y_);
    Mat image;
    if (show_step_times_) start = clock();
    resize(original_image, image, resize_dimensions);
    if (show_step_times_) duration_resize = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;

    cvtColor(image, img_HSV, CV_BGR2HSV); // Convert the captured frame from BGR to HSV
    if (show_step_times_) duration_cvtColor = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;


    Mat imgThresholded;
    if (show_step_times_) start = clock();
    inRange(img_HSV, Scalar(low_hue_range_, low_saturation_range_, low_value_range_), Scalar(high_hue_range_, high_saturation_range_, high_value_range_), imgThresholded); //Threshold the image
    if (show_step_times_) duration_inRange = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;

    if (do_erode_dilate_) {
        if (show_step_times_) start = clock();
        // Remove small objects from the background.
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
        dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
        if (show_step_times_) duration_erode_dilate_ = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;
    }

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    double contourSize;
    Mat tempImage;

    if (show_step_times_) start = clock();
    imgThresholded.copyTo(tempImage);
    if (show_step_times_) duration_copyTo = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;

    if (show_step_times_) start = clock();
    findContours(tempImage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    if (show_step_times_) duration_findContours = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;

    if (show_step_times_) start = clock();
    vector<Rect> boundRect( contours.size() );
    vector<vector<Point> > contours_poly( contours.size() );
    if (show_step_times_) duration_contoursPoly = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;

    Point2f center;
    //ROS_INFO("Found %ld contours", contours.size());
    float radius;

    if (show_step_times_) start = clock();
    object_detected_ = false;
    if (!contours.empty()) {
        // Find largest blob.
        size_t max_blob_index = -1;
        int max_blob_size = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            contourSize = contourArea(contours[i]);
            if ((contourSize > max_blob_size) && (contourSize >= low_contour_area_) && (contourSize <= high_contour_area_)) {
                max_blob_index = i;
                max_blob_size = contourSize;
            }
        }

        if (max_blob_index != -1) {
            approxPolyDP( Mat(contours[max_blob_index]), contours_poly[max_blob_index], 3, true );
            //boundRect[max_blob_index] = boundingRect( Mat(contours_poly[max_blob_index]) );
            minEnclosingCircle( (Mat) contours_poly[max_blob_index], center, radius );

            object_x_ = center.x;
            object_y_ = center.y;

            image_width_ = image.cols;
            image_height_ = image.rows;
            object_detected_ = true;
            object_area_ = max_blob_size;

            if (show_debug_windows_) {
                Scalar color = Scalar(0, 0, 255);
                Scalar non_primary_color = Scalar(0, 255, 255);
                circle(image, center, (int)radius, color, 5, 8, 0 );
                for (size_t i = 0; i < contours.size(); i++) {
                    if (i == max_blob_index) continue;
                    approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
                    //boundRect[i] = boundingRect( Mat(contours_poly[i]) );
                    float radius;
                    Point2f obj_center;
                    minEnclosingCircle( (Mat) contours_poly[i], obj_center, radius );
                    circle(image, obj_center, (int) radius, non_primary_color, 4, 8, 0);
                    if (ll_annotation_.length() > 0) {
                        putText(image, ll_annotation_, cvPoint(4, image.rows - 4), FONT_HERSHEY_DUPLEX, 2.0, ll_color_, 8, true);
                    }

                    if (lr_annotation_.length() > 0) {
                        putText(image, lr_annotation_, cvPoint(image.cols / 2 + 4, image.rows - 4), FONT_HERSHEY_DUPLEX, 2.0, lr_color_, 8, true);
                    }

                    if (ul_annotation_.length() > 0) {
                        putText(image, ul_annotation_, cvPoint(4, 50), FONT_HERSHEY_DUPLEX, 2.0, ul_color_, 8, false);
                    }

                    if (ur_annotation_.length() > 0) {
                        putText(image, ur_annotation_, cvPoint(image.cols / 2 + 4, 50), FONT_HERSHEY_DUPLEX, 2.0, ur_color_, 8, false);
                    }
                }

                sensor_msgs::ImagePtr annotated_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
                image_pub_annotated_.publish(annotated_image);
            }

            stringstream msg;
            msg << "ConeDetector:Found;X:" << object_x_
                << ";Y:" << object_y_
                << ";AREA:" << max_blob_size
                << ";I:" << max_blob_index
                << ";ROWS:" << image_height_
                << ";COLS:" << image_width_;
            std_msgs::String message;
            message.data = msg.str();
            cone_found_pub_.publish(message);
        }

        if (show_step_times_) duration_find_argest = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;

        stringstream msg;
        msg << "ConeDetector:NotFound;X:0;Y:0;AREA:0;I:0;ROWS:"
            << image_height_
            << ";COLS:" << image_width_;
        std_msgs::String message;
        message.data = msg.str();
        cone_found_pub_.publish(message);
        duration_find_argest = 0;
    } else {
        if (show_debug_windows_) {
            sensor_msgs::ImagePtr annotated_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
            image_pub_annotated_.publish(annotated_image);
        }
    }

    if (show_step_times_) start = clock();
    if (show_debug_windows_) {
        sensor_msgs::ImagePtr thresholded_image = cv_bridge::CvImage(std_msgs::Header(), "mono8", imgThresholded).toImageMsg();
        image_pub_thresholded_.publish(thresholded_image);
    }

    duration_imshow = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;
    if (show_step_times_) ROS_INFO("durations resize: %7.5f, "
                                   "cvtColor: %7.5f"
                                   ", inRange: %7.5f"
                                   ", erode/dilate: %7.5f"
                                   ", findLargest: %7.5f"
                                   ", showWindows: %7.5f"
                                   ", copyTo: %7.5f"
                                   ", findContours: %7.5f"
                                   ", contoursPoly: %7.5f",
                                       duration_resize,
                                       duration_cvtColor,
                                       duration_inRange,
                                       duration_erode_dilate_,
                                       duration_find_argest,
                                       duration_imshow,
                                       duration_copyTo,
                                       duration_findContours, duration_contoursPoly);
    if (show_debug_windows_) { waitKey(1); }
}

ConeDetector::ConeDetector() :
    nh_(ros::NodeHandle("~")),
    it_(nh_),
    do_erode_dilate_(true),
    low_hue_range_(0),
    high_hue_range_(44),
    low_saturation_range_(181),
    high_saturation_range_(255),
    low_value_range_(169),
    high_value_range_(255),
    low_contour_area_(500),
    high_contour_area_(200000),
    show_debug_windows_(false),
    show_step_times_(false),
    ll_annotation_(""),
    ll_color_(255, 255, 0),
    lr_annotation_(""),
    lr_color_(255, 255, 255),
    resize_x_(320),
    resize_y_(240),
    ul_annotation_(""),
    ul_color_(255, 255, 255),
    ur_annotation_(""),
    ur_color_(255, 255, 255) {
    //    f = boost::bind(&ConeDetector::configurationCallback, _1, _2);
    //    dynamicConfigurationServer.setCallback(f);


    assert(ros::param::get("~camera_name", camera_name_));
    assert(ros::param::get("~do_erode_dilate", do_erode_dilate_));
    assert(ros::param::get("~image_topic_name", image_topic_name_));
    assert(ros::param::get("~show_debug_windows", show_debug_windows_));
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
    ROS_INFO("[ConeDetector] PARAM do_erode_dilate: %s", do_erode_dilate_ ? "TRUE" : "FALSE");
    ROS_INFO("[ConeDetector] PARAM image_topic_name: %s", image_topic_name_.c_str());
    ROS_INFO("[ConeDetector] PARAM low_contour_area: %d, high_contour_area: %d", low_contour_area_, high_contour_area_);
    ROS_INFO("[ConeDetector] PARAM low_hue_range: %d, high_hue_range: %d", low_hue_range_, high_hue_range_);
    ROS_INFO("[ConeDetector] PARAM low_saturation_range: %d, high_saturation_range: %d, ", low_saturation_range_, high_saturation_range_);
    ROS_INFO("[ConeDetector] PARAM low_value_range: %d, high_value_range: %d", low_value_range_, high_value_range_);
    ROS_INFO("[ConeDetector] PARAM resize_x: %d, resize_y: %d", resize_x_, resize_y_);
    ROS_INFO("[ConeDetector] PARAM show_step_times: %s", show_step_times_ ? "TRUE" : "FALSE");
    ROS_INFO("[ConeDetector] PARAM show_debug_windows: %s", show_debug_windows_ ? "TRUE" : "FALSE");

    //show_debug_windows_ = false;

    annotateService = nh_.advertiseService("annotate_detector_image", &ConeDetector::annotateCb, this);
    image_pub_annotated_ = it_.advertise("cone_detector/annotated_image", 1);
    image_pub_thresholded_ = it_.advertise("cone_detector/thresholded_image", 1);
    image_sub_ = it_.subscribe(image_topic_name_, 1, &ConeDetector::imageTopicCb, this);
    cone_found_pub_ = nh_.advertise<std_msgs::String>("cone_detector_summary", 2);
    if (show_debug_windows_) {
        static const char* controlWindowName = "[kaimi_mid_camera] Control";

        // Create trackbars in "Control" window
        namedWindow(controlWindowName, CV_WINDOW_AUTOSIZE); //create a window called "Control"
        cvCreateTrackbar("LowH", controlWindowName, &low_hue_range_, 179); //Hue (0 - 179)
        cvCreateTrackbar("HighH", controlWindowName, &high_hue_range_, 179);

        cvCreateTrackbar("LowS", controlWindowName, &low_saturation_range_, 255); //Saturation (0 - 255)
        cvCreateTrackbar("HighS", controlWindowName, &high_saturation_range_, 255);

        cvCreateTrackbar("LowV", controlWindowName, &low_value_range_, 255); //Value (0 - 255)
        cvCreateTrackbar("HighV", controlWindowName, &high_value_range_, 255);

        cvCreateTrackbar("low_contour_area", controlWindowName, &low_contour_area_, 200000);
        cvCreateTrackbar("high_contour_area", controlWindowName, &high_contour_area_, 200000);
    }
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
