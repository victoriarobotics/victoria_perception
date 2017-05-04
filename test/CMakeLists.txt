#catkin_add_gtest(${PROJECT_NAME}-test 
if(CATKIN_ENABLE_TESTING)
	find_package(rostest REQUIRED)
	add_rostest_gtest(victoria_perception_kmeans_test
		kmeans_action.test
		kmeans_action_test.cpp
	)
	target_link_libraries(victoria_perception_kmeans_test
	  ${catkin_LIBRARIES}
	  ${OpenCV_LIBS}
	  ${Boost_LIBRARIES}
	)
endif()