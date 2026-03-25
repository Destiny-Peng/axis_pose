import re

with open("src/poseEstimate.cpp", "r") as f:
    content = f.read()

# Add parameter declarations
content = content.replace('this->declare_parameter("use_sacline", use_sacline_);', 
                          'this->declare_parameter("use_sacline", use_sacline_);\n        this->declare_parameter("algorithm_type", algorithm_type_);')

content = content.replace('use_sacline_ = this->get_parameter("use_sacline").as_bool();', 
                          'use_sacline_ = this->get_parameter("use_sacline").as_bool();\n        algorithm_type_ = this->get_parameter("algorithm_type").as_string();')

# Replace the block with algorithm dispatch in syncCallback
old_dispatch = """        // compute pose using valid_cloud
        geometry_msgs::msg::PoseStamped pose_msg;
        if (benchmark_)
        {
            if (use_sacline_)"""

new_dispatch = """        // compute pose using valid_cloud
        geometry_msgs::msg::PoseStamped pose_msg;
        if (benchmark_)
        {
            if (algorithm_type_ == "proposed")
            {
                auto tup = benchmark_->run("pose_proposed", [this, &valid_cloud, &mask_cv, &depth_msg]()
                                           {
                    auto p = this->computePoseProposed(valid_cloud, mask_cv, depth_msg->header.stamp);
                    return std::make_tuple(p, static_cast<size_t>(valid_cloud->size())); }, [](const std::tuple<geometry_msgs::msg::PoseStamped, size_t> &t) -> std::vector<std::string>
                                           {
                    const auto &pose = std::get<0>(t).pose;
                    size_t n = std::get<1>(t);
                    std::vector<std::string> out;
                    out.reserve(8);
                    out.push_back(std::to_string(n));
                    out.push_back(std::to_string(pose.position.x));
                    out.push_back(std::to_string(pose.position.y));
                    out.push_back(std::to_string(pose.position.z));
                    out.push_back(std::to_string(pose.orientation.x));
                    out.push_back(std::to_string(pose.orientation.y));
                    out.push_back(std::to_string(pose.orientation.z));
                    out.push_back(std::to_string(pose.orientation.w));
                    return out; });
                pose_msg = std::get<0>(tup);
            }
            else if (use_sacline_ || algorithm_type_ == "sacline")"""

content = content.replace(old_dispatch, new_dispatch)

old_else_dispatch = """        else
        {
            if (use_sacline_)
                pose_msg = computePoseFromSACLine(valid_cloud, depth_msg->header.stamp);
            else
                pose_msg = computePoseFromCloud(valid_cloud, depth_msg->header.stamp);
        }"""

new_else_dispatch = """        else
        {
            if (algorithm_type_ == "proposed")
                pose_msg = computePoseProposed(valid_cloud, mask_cv, depth_msg->header.stamp);
            else if (use_sacline_ || algorithm_type_ == "sacline")
                pose_msg = computePoseFromSACLine(valid_cloud, depth_msg->header.stamp);
            else
                pose_msg = computePoseFromCloud(valid_cloud, depth_msg->header.stamp);
        }"""

content = content.replace(old_else_dispatch, new_else_dispatch)

# Add computePoseProposed function just before the final bracket "} // namespace axispose"
proposed_func = """
    geometry_msgs::msg::PoseStamped PoseEstimate::computePoseProposed(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const cv::Mat &mask_cv, const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::PoseStamped pose;
        if (!cloud || cloud->empty()) return pose;

        axispose::GaussianMapSolver solver;
        Eigen::Vector3f out_axis, out_point;
        float out_radius = 0.05f;
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
        
        if (!solver.estimateAxis(cloud_ptr, out_axis, out_point, out_radius)) {
            RCLCPP_WARN(this->get_logger(), "GaussianMapSolver failed");
            return pose;
        }

        cv::Mat mask_dt;
        cv::distanceTransform(mask_cv == 0, mask_dt, cv::DIST_L2, 3);
        
        Eigen::Vector3d d = out_axis.cast<double>();
        Eigen::Vector3d p = out_point.cast<double>();
        Eigen::Vector3d m = p.cross(d);

        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0, 0) = fx_; K(1, 1) = fy_; K(0, 2) = cx_; K(1, 2) = cy_;

        axispose::CeresJointOptimizer optimizer;
        optimizer.optimizePose(d, m, out_radius, cloud_ptr, mask_dt, K);

        Eigen::Vector3d optimized_point = d.cross(m); // minimum distance point to origin
        
        pose.pose.position.x = optimized_point.x();
        pose.pose.position.y = optimized_point.y();
        pose.pose.position.z = optimized_point.z();

        Eigen::Vector3d axis_z = d;
        Eigen::Vector3d up = Eigen::Vector3d::UnitY();
        if (std::abs(axis_z.dot(up)) > 0.99) up = Eigen::Vector3d::UnitX();
        Eigen::Vector3d axis_x = up.cross(axis_z).normalized();
        Eigen::Vector3d axis_y = axis_z.cross(axis_x).normalized();
        
        Eigen::Matrix3d R;
        R.col(0) = axis_x;
        R.col(1) = axis_y;
        R.col(2) = axis_z;
        Eigen::Quaterniond q(R);
        q.normalize();
        
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();

        return pose;
    }
"""

content = re.sub(r'\}[\s]*// namespace axispose', proposed_func + '\n} // namespace axispose', content)

with open("src/poseEstimate.cpp", "w") as f:
    f.write(content)
