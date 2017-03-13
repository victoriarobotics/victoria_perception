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

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>

int main(int argc, char** argv) {
    ros::init(argc, argv, "publish_image_node");
    ros::NodeHandle nh;
    std::string image_path;
    std::string image_topic_name;

    ros::param::get("~image_path", image_path);
    ros::param::get("~image_topic_name", image_topic_name);
    ROS_INFO("[publish_image_node] image_path: %s", image_path.c_str());
    ROS_INFO("[publish_image_node] image_topic_name: %s", image_topic_name.c_str());
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise(image_topic_name, 1);
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    cv::waitKey(30);
    if (image.empty()) {
        ROS_INFO("[publish_image_node] unable to load image: \"%s\"", image_path.c_str());
        exit(-1);
    }

    ROS_INFO("[publish_image_node] image size cols: %d, rows: %d", image.cols, image.rows);

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

    ros::Rate loop_rate(5);
    while (nh.ok()) {
        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
}
