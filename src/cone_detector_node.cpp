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

#include <ros/ros.h>
#include <ros/console.h>

#include "cone_detector/cone_detector.h"
#include "victoria_perception/ObjectDetector.h"

int main(int argc, char** argv) {
    ConeDetector* coneDetector;

    ros::init(argc, argv, "cone_detector_node");

    int fps;
	ros::NodeHandle nh;
    victoria_perception::ObjectDetector detector_msg;

    ros::param::get("~fps", fps);
    ROS_INFO("[cone_detector_node] PARAM fps: %d", fps);
    coneDetector = &ConeDetector::singleton();

	ros::Publisher detector_pub = nh.advertise<victoria_perception::ObjectDetector>("cone_detector", 2);
    ros::Rate rate(fps);

    while (ros::ok()) {
    	detector_msg.header.stamp = ros::Time::now();
		detector_msg.image_height = coneDetector->imageHeight();
		detector_msg.image_width = coneDetector->imageWidth();
		detector_msg.object_area = coneDetector->objectArea();
		detector_msg.object_detected = coneDetector->objectDetected();
		detector_msg.object_x = coneDetector->objectX();
		detector_msg.object_y = coneDetector->objectY();
		detector_pub.publish(detector_msg);
        rate.sleep();
        ros::spinOnce();
    }

    ros::spin();
    return 0;
}
