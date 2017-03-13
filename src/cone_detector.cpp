#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <sys/stat.h>

#include <camera_info_manager/camera_info_manager.h>

#include "cone_detector/cone_detector.h"
#include "victoria_perception/ObjectDetector.h"

using namespace cv;
using namespace std;

void ConeDetector::imageCb(Mat& image) {
    clock_t start;
    double duration_cvtColor = 0;       // Time taken for cvtColor call.
    double duration_inRange = 0;        // Time taken for inRange call.
    double duration_copyTo = 0;         // Time taken for copyTo operation.
    double duration_find_argest = 0;    // Time taken to find largets blob.
    double duration_imshow = 0;         // Time taken for imshow calls
    double duration_findContours = 0;   // Time taken for findContours call.
    double duration_contoursPoly = 0;   // Time taken for contoursPoly call.

    Mat img_HSV;
    if (show_step_times_) start = clock();
    cvtColor(image, img_HSV, CV_BGR2HSV); // Convert the captured frame from BGR to HSV
    if (show_step_times_) duration_cvtColor = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;


    Mat imgThresholded;
    if (show_step_times_) start = clock();
    inRange(img_HSV, Scalar(low_hue_range_, low_saturation_range_, low_value_range_), Scalar(high_hue_range_, high_saturation_range_, high_value_range_), imgThresholded); //Threshold the image
    if (show_step_times_) duration_inRange = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;

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
            boundRect[max_blob_index] = boundingRect( Mat(contours_poly[max_blob_index]) );
            minEnclosingCircle( (Mat) contours_poly[max_blob_index], center, radius );

            object_x_ = center.x;
            object_y_ = center.y;

            image_width_ = image.cols;
            image_height_ = image.rows;
            object_detected_ = true;
            object_area_ = max_blob_size;

            if (show_debug_windows_) {
                Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
                //#####               circle(image, center, (int)radius, color, 2, 8, 0 );
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
            //ROS_INFO("[ConeDetector::imageCb] FOUND at x: %d, y: %d, area: %d", object_x_, object_y_, max_blob_size);
        }

        if (show_step_times_) duration_find_argest = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;
    } else {
        stringstream msg;
        msg << "ConeDetector:NotFound;X:0;Y:0;AREA:0;I:0;ROWS:"
            << image_height_
            << ";COLS:" << image_width_;
        std_msgs::String message;
        message.data = msg.str();
        cone_found_pub_.publish(message);
        //ROS_INFO("[ConeDetector::imageCb] NOT FOUND");
        duration_find_argest = 0;
    }

    if (show_step_times_) start = clock();
    if (show_debug_windows_) {
        imshow("[kaimi_mid_camera] Raw Image", image); //show the original image
        imshow("[kaimi_mid_camera] Thresholded Image", imgThresholded); //show the thresholded image
        // ROS_INFO("[ConeDetector::imageCb] showed images");
        cv::waitKey(25);
    }

    duration_imshow = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;
    if (show_step_times_) ROS_INFO("durations cvtColor: %7.5f, inRange: %7.5f, findLargest: %7.5f, showWindows: %7.5f, copyTo: %7.5f, findContours: %7.5f, contoursPoly: %7.5f",
                                 duration_cvtColor,
                                 duration_inRange,
                                 duration_find_argest,
                                 duration_imshow,
                                 duration_copyTo,
                                 duration_findContours, duration_contoursPoly);
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
    show_debug_windows_(false),
    show_step_times_(false) {
    //    f = boost::bind(&ConeDetector::configurationCallback, _1, _2);
    //    dynamicConfigurationServer.setCallback(f);


    ros::param::get("camera_name", camera_name_);
    ros::param::get("~image_topic_name", image_topic_name_);
    ros::param::get("~show_debug_windows", show_debug_windows_);
    ros::param::get("~show_step_times", show_step_times_);

    ros::param::get("low_hue_range", low_hue_range_);
    ros::param::get("high_hue_range", high_hue_range_);
    ros::param::get("low_saturation_range", low_saturation_range_);
    ros::param::get("high_saturation_range", high_saturation_range_);
    ros::param::get("low_value_range", low_value_range_);
    ros::param::get("high_value_range", high_value_range_);
    ros::param::get("low_contour_area", low_contour_area_);
    ros::param::get("high_contour_area", high_contour_area_);

    ROS_INFO("[ConeDetector] PARAM camera_name: %s", camera_name_.c_str());
    ROS_INFO("[ConeDetector] PARAM image_topic_name: %s", image_topic_name_.c_str());
    ROS_INFO("[ConeDetector] PARAM show_windows: %d", show_debug_windows_);
    ROS_INFO("[ConeDetector] PARAM low_hue_range: %d, high_hue_range: %d", low_hue_range_, high_hue_range_);
    ROS_INFO("[ConeDetector] PARAM low_saturation_range: %d, high_saturation_range: %d, ", low_saturation_range_, high_saturation_range_);
    ROS_INFO("[ConeDetector] PARAM low_value_range: %d, high_value_range: %d", low_value_range_, high_value_range_);
    ROS_INFO("[ConeDetector] PARAM low_contour_area: %d, high_contour_area: %d", low_contour_area_, high_contour_area_);

    //show_debug_windows_ = false;

    image_sub_ = it_.subscribe(image_topic_name_, 1, &ConeDetector::imageTopicCb, this);
    cone_found_pub_ = nh_.advertise<std_msgs::String>("cone_detector_summary", 2);
    if (show_debug_windows_) {
        static const char* controlWindowName = "[kaimi_mid_camera] Control";

        namedWindow("[kaimi_mid_camera] Raw Image", WINDOW_NORMAL);
        namedWindow("[kaimi_mid_camera] Thresholded Image", WINDOW_NORMAL);

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
