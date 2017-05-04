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
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include "victoria_perception/KmeansAction.h"

void feedbackCb(const victoria_perception::KmeansFeedbackConstPtr& feedback) {
    ROS_INFO("[invoke_kmeans_action_node] Feedback: %s", feedback->step.c_str());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "kmeans_action_node");

    // Create the action client.
    // Last param = true causes the client to spin its own thread.
    actionlib::SimpleActionClient<victoria_perception::KmeansAction> ac("compute_kmeans", true);

    ROS_INFO("[invoke_kmeans_action_node] Waiting for action server to start.");
    // wait for the action server to start
    ac.waitForServer(); //will wait for infinite time

    ROS_INFO("[invoke_kmeans_action_node] Action server started, sending goal.");
    // send a goal to the action
    victoria_perception::KmeansGoal goal;
    goal.attempts = 1;
    goal.image_topic_name = "/usb_cam/image_raw";
    goal.number_clusters = 16;
    goal.resize_width = 320;
    ac.sendGoal(goal, NULL, NULL, feedbackCb);

    //wait for the action to return
    bool finished_before_timeout = ac.waitForResult(ros::Duration(5.0));

    if (finished_before_timeout) {
    	ROS_INFO("[invoke_kmeans_action_node] state: %s", ac.getState().toString().c_str());
    	ROS_INFO("[invoke_kmeans_action_node] result msg: %s", ac.getResult()->result_msg.c_str());
    	ROS_INFO("[invoke_kmeans_action_node] kmeans result: %s", ac.getResult()->kmeans_result.c_str());
    } else {
        ROS_INFO("Action did not finish before the time out.");
    }
    ros::spin();
    
    return 0;
}
