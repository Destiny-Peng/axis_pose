sed -i 's/this->declare_parameter("use_sacline", use_sacline_);/this->declare_parameter("use_sacline", use_sacline_);\n        this->declare_parameter("algorithm_type", algorithm_type_);/g' src/poseEstimate.cpp
sed -i 's/use_sacline_ = this->get_parameter("use_sacline").as_bool();/use_sacline_ = this->get_parameter("use_sacline").as_bool();\n        algorithm_type_ = this->get_parameter("algorithm_type").as_string();/g' src/poseEstimate.cpp

awk '
/geometry_msgs::msg::PoseStamped pose_msg;/ {
    print "        geometry_msgs::msg::PoseStamped pose_msg;\n        if (benchmark_)\n        {\n            if (algorithm_type_ == \"proposed\")\n            {\n                auto tup = benchmark_->run(\"pose_proposed\", [this, &valid_cloud, &mask_cv, &depth_msg]()\n                                           {\n                    auto p = this->computePoseProposed(valid_cloud, mask_cv, depth_msg->header.stamp);\n                    return std::make_tuple(p, static_cast<size_t>(valid_cloud->size())); }, [](const std::tuple<geometry_msgs::msg::PoseStamped, size_t> &t) -> std::vector<std::string>\n                                           {\n                    const auto &pose = std::get<0>(t).pose;\n                    size_t n = std::get<1>(t);\n                    std::vector<std::string> out;\n                    out.reserve(8);\n                    out.push_back(std::to_string(n));\n                    out.push_back(std::to_string(pose.position.x));\n                    out.push_back(std::to_string(pose.position.y));\n                    out.push_back(std::to_string(pose.position.z));\n                    out.push_back(std::to_string(pose.orientation.x));\n                    out.push_back(std::to_string(pose.orientation.y));\n                    out.push_back(std::to_string(pose.orientation.z));\n                    out.push_back(std::to_string(pose.orientation.w));\n                    return out; });\n                pose_msg = std::get<0>(tup);\n            }\n            else if (use_sacline_ || algorithm_type_ == \"sacline\")"
    flag = 1
    next
}
/if (use_sacline_)/ && flag == 1 {
    flag = 2
    next
}
/else if (use_sacline_)/ && flag == 3 {
    print "            else if (use_sacline_ || algorithm_type_ == \"sacline\")"
    next
}
/else\n        {/ {
    print "        else\n        {\n            if (algorithm_type_ == \"proposed\")\n                pose_msg = computePoseProposed(valid_cloud, mask_cv, depth_msg->header.stamp);"
    flag = 3
    next
}
{ print }
' src/poseEstimate.cpp > src/poseEstimate_temp.cpp
mv src/poseEstimate_temp.cpp src/poseEstimate.cpp
