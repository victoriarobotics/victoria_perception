#include <ros/ros.h>
#include <ros/console.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include "victoria_perception/KmeansAction.h"

// Bring in gtest
#include <gtest/gtest.h>

// There is an image, test_raster.jpg, which was hand created to have 16 columns of solid colors.
// The kmeans analysis should produce a result set that accurately discovers those 16 partitions.
// If you look at the image, you should see something like:
// clust			 bar#
//  0  60, 255, 128  9-brn
//  1 165, 255, 255 11-pink
//  2   0, 255, 254  6-red
//  3  60, 255, 255  2-green
//  4  15, 170, 153  0-brown
//  5 120, 255, 254 10-blue
//  6 150, 255, 255  3-pink
//  7 150, 255, 126  5-purple
//  8  80, 153, 254 13-cyan
//  9 120, 154, 254 14-lt blue
// 10  90, 255, 255  1-cyan
// 11  15, 255, 255  4-orange
// 12  90, 255, 129 15-green
// 13   0, 255, 128  8-brown
// 14  30, 153, 255 12-yellow
// 15  30, 255, 255  7-yellow
//
// clust is the cluster number is the result set. Bar# is the corresponding column number in 
// the test image.
//
// Which corresponds to the expected result set.

static const std::string G_EXPECTED_RESULT =
"{cluster=0;min_hue=60;max_hue=60;min_saturation=255;max_saturation=255;min_value=128;max_value=128;pixels=4800}"
"{cluster=1;min_hue=165;max_hue=165;min_saturation=255;max_saturation=255;min_value=255;max_value=255;pixels=4800}"
"{cluster=2;min_hue=0;max_hue=0;min_saturation=255;max_saturation=255;min_value=254;max_value=254;pixels=4800}"
"{cluster=3;min_hue=60;max_hue=60;min_saturation=255;max_saturation=255;min_value=255;max_value=255;pixels=4800}"
"{cluster=4;min_hue=15;max_hue=15;min_saturation=170;max_saturation=170;min_value=153;max_value=153;pixels=4800}"
"{cluster=5;min_hue=120;max_hue=120;min_saturation=255;max_saturation=255;min_value=254;max_value=254;pixels=4800}"
"{cluster=6;min_hue=150;max_hue=150;min_saturation=255;max_saturation=255;min_value=255;max_value=255;pixels=4800}"
"{cluster=7;min_hue=150;max_hue=150;min_saturation=255;max_saturation=255;min_value=126;max_value=126;pixels=4800}"
"{cluster=8;min_hue=80;max_hue=80;min_saturation=153;max_saturation=153;min_value=254;max_value=254;pixels=4800}"
"{cluster=9;min_hue=120;max_hue=120;min_saturation=154;max_saturation=154;min_value=254;max_value=254;pixels=4800}"
"{cluster=10;min_hue=90;max_hue=90;min_saturation=255;max_saturation=255;min_value=255;max_value=255;pixels=4800}"
"{cluster=11;min_hue=15;max_hue=15;min_saturation=255;max_saturation=255;min_value=255;max_value=255;pixels=4800}"
"{cluster=12;min_hue=90;max_hue=90;min_saturation=255;max_saturation=255;min_value=129;max_value=129;pixels=4800}"
"{cluster=13;min_hue=0;max_hue=0;min_saturation=255;max_saturation=255;min_value=128;max_value=128;pixels=4800}"
"{cluster=14;min_hue=30;max_hue=30;min_saturation=153;max_saturation=153;min_value=255;max_value=255;pixels=4800}"
"{cluster=15;min_hue=30;max_hue=30;min_saturation=255;max_saturation=255;min_value=255;max_value=255;pixels=4800}";

// Declare a test
TEST(TestSuite, testCase1)
{
    // create the action client
    // true causes the client to spin its own thread
    actionlib::SimpleActionClient<victoria_perception::KmeansAction> ac("compute_kmeans", true);

    // wait for the action server to start
    ac.waitForServer(); //will wait for infinite time

    // send a goal to the action
    victoria_perception::KmeansGoal goal;
    goal.attempts = 1;
    goal.image_topic_name = "/usb_cam/image_raw";
    goal.number_clusters = 16;
    goal.resize_width = 320;
    ac.sendGoal(goal);

    //wait for the action to return
    bool finished_before_timeout = ac.waitForResult(ros::Duration(7.0));

    if (finished_before_timeout) {
    	EXPECT_STREQ("SUCCEEDED", ac.getState().toString().c_str());
    	EXPECT_STREQ("OK", ac.getResult()->result_msg.c_str());
    	EXPECT_STREQ(G_EXPECTED_RESULT.c_str(), ac.getResult()->kmeans_result.c_str());
    } else {
        ADD_FAILURE() << "Action did not finish before the time out.";
    }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "test");
  ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}