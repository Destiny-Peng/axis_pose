// AlgorithmBenchmark: lightweight benchmarking and CSV logger
// C++17
#ifndef AXISPOSE_BENCHMARK_HPP_
#define AXISPOSE_BENCHMARK_HPP_

#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <exception>
#include <optional>
#include <filesystem>

namespace axispose
{

    class AlgorithmBenchmark
    {
    public:
        // output_dir: directory to place CSV files (will be created if missing)
        // csv_name: CSV filename inside output_dir (default: metrics.csv)
        // enabled: when false, run() has near-zero overhead (no logging)
        // extra_headers: names for formatter-provided columns
        AlgorithmBenchmark(const std::string &output_dir, const std::string &csv_name = "metrics.csv", bool enabled = true, const std::vector<std::string> &extra_headers = {})
            : enabled_(enabled), extra_headers_(extra_headers)
        {
            if (!enabled_)
                return;
            try
            {
                std::filesystem::path outp = std::filesystem::absolute(std::filesystem::path(output_dir));
                if (!std::filesystem::exists(outp))
                {
                    std::filesystem::create_directories(outp);
                }
                auto csv_path = (outp / csv_name);
                bool write_header = true;
                if (std::filesystem::exists(csv_path))
                {
                    try
                    {
                        if (std::filesystem::file_size(csv_path) > 0)
                            write_header = false;
                    }
                    catch (...)
                    { /* best effort */
                    }
                }
                ofs_.open(csv_path.string(), std::ofstream::out | std::ofstream::app);
                if (!ofs_)
                    throw std::runtime_error("AlgorithmBenchmark: cannot open file " + csv_path.string());
                // write header only when file is new/empty
                if (write_header)
                {
                    ofs_ << "run_id,name,timestamp_iso,elapsed_ms";
                    for (const auto &h : extra_headers_)
                        ofs_ << "," << escape_csv_field(h);
                    ofs_ << '\n';
                    ofs_.flush();
                }
            }
            catch (const std::exception &e)
            {
                throw;
            }
        }

        ~AlgorithmBenchmark()
        {
            if (ofs_.is_open())
            {
                ofs_.flush();
                ofs_.close();
            }
        }

        // Non-void return type
        template <typename Func, typename LogFunc>
        auto run(const std::string &name, Func &&algo, LogFunc &&formatter)
            -> std::decay_t<typename std::invoke_result_t<Func>>
        {
            using RetT = std::decay_t<typename std::invoke_result_t<Func>>;
            if (!enabled_)
            {
                return algo();
            }

            const auto run_id = next_run_id();
            const auto ts = now_iso();

            auto t0 = Clock::now();
            std::optional<RetT> result;
            std::exception_ptr eptr = nullptr;
            try
            {
                result = algo();
            }
            catch (...)
            {
                eptr = std::current_exception();
            }
            auto t1 = Clock::now();
            double elapsed_ms = ms_duration(t0, t1);

            // prepare formatted fields
            std::vector<std::string> fields;
            if (!eptr)
            {
                try
                {
                    fields = formatter(*result);
                }
                catch (...)
                {
                    // formatting failure: record placeholder
                    fields = {"<format_error>"};
                }
            }
            else
            {
                fields = {"<exception>"};
            }

            write_row(run_id, name, ts, elapsed_ms, fields);

            if (eptr)
                std::rethrow_exception(eptr);

            return *result;
        }

        // Void-returning overload: formatter must be callable with no args returning vector<string>
        template <typename Func, typename LogFunc>
        auto run(const std::string &name, Func &&algo, LogFunc &&formatter)
            -> std::enable_if_t<std::is_void<typename std::invoke_result_t<Func>>::value, void>
        {
            if (!enabled_)
            {
                algo();
                return;
            }

            const auto run_id = next_run_id();
            const auto ts = now_iso();
            auto t0 = Clock::now();
            std::exception_ptr eptr = nullptr;
            try
            {
                algo();
            }
            catch (...)
            {
                eptr = std::current_exception();
            }
            auto t1 = Clock::now();
            double elapsed_ms = ms_duration(t0, t1);

            std::vector<std::string> fields;
            if (!eptr)
            {
                try
                {
                    fields = formatter();
                }
                catch (...)
                {
                    fields = {"<format_error>"};
                }
            }
            else
            {
                fields = {"<exception>"};
            }

            write_row(run_id, name, ts, elapsed_ms, fields);

            if (eptr)
                std::rethrow_exception(eptr);
        }

        // enable/disable at runtime
        void set_enabled(bool e)
        {
            enabled_ = e;
        }

    private:
        using Clock = std::chrono::steady_clock;

        std::ofstream ofs_;
        bool enabled_ = true;
        std::mutex mtx_;
        std::uint64_t run_counter_ = 0;
        std::vector<std::string> extra_headers_;

        static std::string now_iso()
        {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &t);
#else
            localtime_r(&t, &tm);
#endif
            std::ostringstream ss;
            ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            ss << "." << std::setfill('0') << std::setw(3) << ms.count();
            return ss.str();
        }

        static double ms_duration(const Clock::time_point &a, const Clock::time_point &b)
        {
            return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
        }

        static std::string to_string_double(double v, int precision = 6)
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(precision) << v;
            return ss.str();
        }

        static std::string escape_csv_field(const std::string &s)
        {
            bool need_quotes = s.find(',') != std::string::npos || s.find('"') != std::string::npos || s.find('\n') != std::string::npos;
            if (!need_quotes)
                return s;
            std::string out = "\"";
            for (char c : s)
            {
                if (c == '"')
                    out += '"', out += '"';
                else
                    out += c;
            }
            out += '"';
            return out;
        }

        void write_row(std::uint64_t run_id, const std::string &name, const std::string &ts, double elapsed_ms, const std::vector<std::string> &fields)
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!ofs_.is_open())
                return;
            ofs_ << run_id << "," << escape_csv_field(name) << "," << ts << "," << to_string_double(elapsed_ms);
            for (const auto &f : fields)
            {
                ofs_ << "," << escape_csv_field(f);
            }
            ofs_ << '\n';
            ofs_.flush();
        }

        std::uint64_t next_run_id()
        {
            std::lock_guard<std::mutex> lk(mtx_);
            return ++run_counter_;
        }
    };

} // namespace axispose

#endif // AXISPOSE_BENCHMARK_HPP_
