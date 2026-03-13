#ifndef AXISPOSE_DEBUG_MANAGER_HPP_
#define AXISPOSE_DEBUG_MANAGER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>

namespace axispose
{

    class DebugManager
    {
    public:
        explicit DebugManager(rclcpp::Node *node) : node_(node)
        {
            // declare parameter and load initial flags
            node_->declare_parameter("debug_flags", std::vector<std::string>{});
            updateFromParam();
            // register parameter callback to update flags at runtime
            node_->add_on_set_parameters_callback(
                std::bind(&DebugManager::paramsCallback, this, std::placeholders::_1));
        }

        bool enabled(const std::string &name) const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return flags_.find(name) != flags_.end();
        }

    private:
        rcl_interfaces::msg::SetParametersResult paramsCallback(const std::vector<rclcpp::Parameter> &params)
        {
            for (const auto &p : params)
            {
                if (p.get_name() == "debug_flags")
                {
                    updateFromParam();
                    break;
                }
            }
            rcl_interfaces::msg::SetParametersResult res;
            res.successful = true;
            res.reason = "";
            return res;
        }

        void updateFromParam()
        {
            std::vector<std::string> vec;
            try
            {
                node_->get_parameter("debug_flags", vec);
            }
            catch (...)
            {
            }
            std::lock_guard<std::mutex> lock(mutex_);
            flags_.clear();
            for (const auto &s : vec)
                flags_.insert(s);
        }

        rclcpp::Node *node_;
        mutable std::mutex mutex_;
        std::unordered_set<std::string> flags_;
    };

} // namespace axispose

#endif // AXISPOSE_DEBUG_MANAGER_HPP_
