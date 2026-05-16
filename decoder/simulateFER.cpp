#include "stabilizerCodes.h"
#include "helpers.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace fs = std::filesystem;


// ============================================================
// Epsilon list options
// ============================================================

enum class EpsilonMode
{
    FixedList,
    LinearRange,
    LogRange,
    DecadeLinearRange
};

std::string epsilon_mode_name(EpsilonMode mode)
{
    switch (mode)
    {
        case EpsilonMode::FixedList:
            return "FixedList";
        case EpsilonMode::LinearRange:
            return "LinearRange";
        case EpsilonMode::LogRange:
            return "LogRange";
        case EpsilonMode::DecadeLinearRange:
            return "DecadeLinearRange";
        default:
            return "Unknown";
    }
}
std::vector<double> make_decade_linear_epsilon_list(
    double epsilon_max,
    double epsilon_min)
{
    std::vector<double> epsilons;

    if (epsilon_max < epsilon_min)
    {
        throw std::runtime_error("epsilon_max must be >= epsilon_min.");
    }

    double decade = 0.1;

    while (decade >= epsilon_min - 1e-15)
    {
        double step = decade / 10.0;

        for (int i = 9; i >= 1; --i)
        {
            double ep = i * step;

            if (ep <= epsilon_max + 1e-15 && ep >= epsilon_min - 1e-15)
            {
                epsilons.push_back(ep);
            }
        }

        decade /= 10.0;
    }

    return epsilons;
}
std::vector<double> make_epsilon_list(
    EpsilonMode mode,
    const std::vector<double> &fixed_list,
    double epsilon_max,
    double epsilon_min,
    double epsilon_step)
{
    std::vector<double> epsilons;

    if (mode == EpsilonMode::FixedList)
    {
        return fixed_list;
    }

    if (mode == EpsilonMode::DecadeLinearRange)
    {
        return make_decade_linear_epsilon_list(epsilon_max, epsilon_min);
    }

    if (epsilon_max < epsilon_min)
    {
        throw std::runtime_error("epsilon_max must be >= epsilon_min.");
    }

    if (epsilon_step <= 0.0)
    {
        throw std::runtime_error("epsilon_step must be positive.");
    }

    if (mode == EpsilonMode::LinearRange)
    {
        for (double ep = epsilon_max; ep >= epsilon_min - 1e-15; ep -= epsilon_step)
        {
            epsilons.push_back(ep);
        }
    }
    else if (mode == EpsilonMode::LogRange)
    {
        if (epsilon_step <= 1.0)
        {
            throw std::runtime_error("For LogRange, epsilon_step must be > 1.");
        }

        for (double ep = epsilon_max; ep >= epsilon_min - 1e-15; ep /= epsilon_step)
        {
            epsilons.push_back(ep);
        }
    }

    return epsilons;
}


// ============================================================
// Parameters and results
// ============================================================

struct DecodeParams
{
    // Code parameters
    int n = 128;
    int k = 36;
    int m = 128;

    // 0: flooding BP
    // 1: genie-aided BP
    // 2: CAMEL
    int decoding_option = 0;

    // 0: depolarzing
    // 1: PL
    int noise_model = 1;

    int number_guess = 1;
    int dec_iter_num = 15;

    double ep0 = 0.1;
    double p = 0.1;

    std::uint64_t max_frame_errors = 300;
    std::uint64_t max_decoded_words = 100000000;

    std::uint64_t progress_interval = 10000000;

    EpsilonMode epsilon_mode = EpsilonMode::DecadeLinearRange;

    double epsilon_max = 0.1;
    double epsilon_min = 0.001;
    double epsilon_step = 0.01;

    std::vector<double> fixed_epsilons = {
        0.12, 0.11, 0.10, 0.09,
        0.08, 0.07, 0.06, 0.05,
        0.04, 0.03, 0.02, 0.01
    };

    DecodeParams()
    {
        if (decoding_option == 0)
        {
            number_guess = 0;
        }
    }

    std::vector<double> epsilons() const
    {
        return make_epsilon_list(
            epsilon_mode,
            fixed_epsilons,
            epsilon_max,
            epsilon_min,
            epsilon_step
        );
    }
};


struct DecodeResult
{
    double epsilon = 0.0;
    std::uint64_t frame_errors = 0;
    std::uint64_t decoded_words = 0;
    double frame_error_rate = 0.0;
    double runtime_seconds = 0.0;
};


// ============================================================
// Utility functions
// ============================================================

std::string decoding_option_name(int option)
{
    switch (option)
    {
    case 0:
        return "Flooding BP";
    case 1:
        return "Genie-aided BP";
    case 2:
        return "CAMEL";
    default:
        return "Unknown";
    }
}


std::string noies_model_name(int model)
{
    switch (model)
    {
        case 0:
            return "Depolarizing";
        case 1:
            return "Phenomenological";
        default:
            return "Unknown";
    }
}


std::string noise_model_folder_name(int model)
{
    switch (model)
    {
        case 0:
            return "Depolarizing";
        case 1:
            return "Phenomenological";
        default:
            return "UnknownNoise";
    }
}


std::string decoding_option_folder_name(int option)
{
    switch (option)
    {
        case 0:
            return "FloodingBP";
        case 1:
            return "GenieAidedBP";
        case 2:
            return "CAMEL";
        default:
            return "UnknownDecoder";
    }
}


std::string result_folder_name(int option, int model)
{
    return noise_model_folder_name(model) + "_" + decoding_option_folder_name(option);
}
std::string p_folder_name(double p)
{
    std::ostringstream ss;
    ss << "p=" << std::setprecision(4) << p;
    return ss.str();
}

fs::path make_output_dir(const DecodeParams &params)
{
    fs::path output_dir =
        fs::path("./decoding_results") /
        (std::to_string(params.n) + "_" + std::to_string(params.k)) /
        result_folder_name(params.decoding_option, params.noise_model);

    if (params.noise_model != 0)
    {
        output_dir /= p_folder_name(params.p);
    }

    fs::create_directories(output_dir);

    return output_dir;
}


void log_line(std::ostream &log_file, const std::string &message)
{
    std::cout << message << std::endl;
    log_file << message << std::endl;
    log_file.flush();
}


// ============================================================
// File writing
// ============================================================

void write_params_json(const DecodeParams &params, const fs::path &path)
{
    const std::vector<double> epsilon_list = params.epsilons();

    std::ofstream file(path);

    file << std::setprecision(17);
    file << "{\n";
    file << "  \"n\": " << params.n << ",\n";
    file << "  \"k\": " << params.k << ",\n";
    file << "  \"m\": " << params.m << ",\n";

    file << "  \"decoding_option\": " << params.decoding_option << ",\n";
    file << "  \"decoding_option_name\": \"" << decoding_option_name(params.decoding_option) << "\",\n";
    file << "  \"noise_model\": \"" <<noise_model_folder_name(params.noise_model) << "\",\n";

    file << "  \"number_guess\": " << params.number_guess << ",\n";
    file << "  \"dec_iter_num\": " << params.dec_iter_num << ",\n";
    file << "  \"ep0\": " << params.ep0 << ",\n";

    file << "  \"max_frame_errors\": " << params.max_frame_errors << ",\n";
    file << "  \"max_decoded_words\": " << params.max_decoded_words << ",\n";
    file << "  \"progress_interval\": " << params.progress_interval << ",\n";
    file << "  \"omp_max_threads\": " << omp_get_max_threads() << ",\n";

    file << "  \"epsilon_mode\": \"" << epsilon_mode_name(params.epsilon_mode) << "\",\n";
    file << "  \"epsilon_max\": " << params.epsilon_max << ",\n";
    file << "  \"epsilon_min\": " << params.epsilon_min << ",\n";
    file << "  \"epsilon_step\": " << params.epsilon_step << ",\n";

    file << "  \"epsilons\": [";
    for (std::size_t i = 0; i < epsilon_list.size(); ++i)
    {
        if (i > 0)
        {
            file << ", ";
        }
        file << epsilon_list[i];
    }
    file << "]\n";

    file << "}\n";
}


void initialize_results_csv(const fs::path &path)
{
    std::ofstream file(path);

    file << "epsilon,"
         << "frame_errors,"
         << "decoded_words,"
         << "frame_error_rate,"
         << "runtime_seconds\n";
}


void append_result_csv(const DecodeResult &result, const fs::path &path)
{
    std::ofstream file(path, std::ios::app);

    file << std::setprecision(17)
         << result.epsilon << ","
         << result.frame_errors << ","
         << result.decoded_words << ","
         << result.frame_error_rate << ","
         << result.runtime_seconds << "\n";
}


// ============================================================
// Decoding simulation
// ============================================================

DecodeResult run_single_epsilon(
    double epsilon,
    const DecodeParams &params,
    fileReader &matrix_supplier,
    std::ostream &log_file)
{
    std::uint64_t started_words = 0;
    std::uint64_t decoded_words = 0;
    std::uint64_t frame_errors = 0;

    const auto start_time = std::chrono::steady_clock::now();

#pragma omp parallel
    {
        while (true)
        {
            bool run_trial = false;

#pragma omp critical(simulation_counters)
            {
                if (started_words < params.max_decoded_words &&
                    frame_errors < params.max_frame_errors)
                {
                    ++started_words;
                    run_trial = true;
                }
            }

            if (!run_trial)
            {
                break;
            }

            stabilizerCodes code(params.n, params.k, params.m, matrix_supplier);

            code.DecodingOption = params.decoding_option;

            code.add_error_given_epsilon(epsilon);

            const bool success = code.decode(
                params.dec_iter_num,
                params.ep0,
                params.number_guess,
                params.p
            );

#pragma omp critical(simulation_counters)
            {
                ++decoded_words;

                if (!success)
                {
                    ++frame_errors;
                }

                if (params.progress_interval > 0 &&
                    decoded_words % params.progress_interval == 0)
                {
                    std::ostringstream msg;
                    msg << "% epsilon=" << epsilon
                        << ", FE " << frame_errors
                        << ", total dec. " << decoded_words
                        << "\\\\";
                    log_line(log_file, msg.str());
                }
            }
        }
    }

    const auto end_time = std::chrono::steady_clock::now();

    const double runtime_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    DecodeResult result;
    result.epsilon = epsilon;
    result.frame_errors = frame_errors;
    result.decoded_words = decoded_words;
    result.frame_error_rate =
        decoded_words == 0
            ? 0.0
            : static_cast<double>(frame_errors) / static_cast<double>(decoded_words);
    result.runtime_seconds = runtime_seconds;

    return result;
}


// ============================================================
// Main
// ============================================================

int main()
{
    DecodeParams params;

    const std::vector<double> epsilon_list = params.epsilons();

    const fs::path output_dir = make_output_dir(params);
    const fs::path params_path = output_dir / "params.json";
    const fs::path results_path = output_dir / "results.csv";
    const fs::path log_path = output_dir / "run.log";

    std::ofstream log_file(log_path);

    write_params_json(params, params_path);
    initialize_results_csv(results_path);

    fileReader matrix_supplier(params.n, params.k, params.m);
    matrix_supplier.check_symplectic();

    log_line(log_file, "% Output directory: " + output_dir.string());
    log_line(log_file, "% Parameters saved to: " + params_path.string());
    log_line(log_file, "% Results saved to: " + results_path.string());

    {
        std::ostringstream msg;
        msg << "% collect " << params.max_frame_errors
            << " frame errors or " << params.max_decoded_words
            << " decoded error patterns";
        log_line(log_file, msg.str());
    }

    {
        std::ostringstream msg;
        msg << "% epsilon mode: " << epsilon_mode_name(params.epsilon_mode);
        log_line(log_file, msg.str());
    }

    {
        std::ostringstream msg;
        msg << "% depolarizing prob. list to be simulated: ";
        for (double ep : epsilon_list)
        {
            msg << ep << ", ";
        }
        log_line(log_file, msg.str());
    }

    log_line(log_file, "% Decoding option: " + decoding_option_name(params.decoding_option));
    log_line(
     log_file,
     "% Noise model: " + noies_model_name(params.noise_model) +
     ", p=" + std::to_string(params.p)
    );

    {
        std::ostringstream msg;
        msg << "% [[" << params.n << "," << params.k << "]], "
            << params.m << " checks, "
            << params.dec_iter_num << " iter, "
            << "ep0=" << params.ep0;
        log_line(log_file, msg.str());
    }

    for (double epsilon : epsilon_list)
    {
        DecodeResult result = run_single_epsilon(
            epsilon,
            params,
            matrix_supplier,
            log_file
        );

        append_result_csv(result, results_path);

        {
            std::ostringstream msg;
            msg << "% FE " << result.frame_errors
                << ", total dec. " << result.decoded_words
                << ", runtime " << result.runtime_seconds
                << " s"
                << "\\\\";
            log_line(log_file, msg.str());
        }

        {
            std::ostringstream msg;
            msg << std::setprecision(17)
                << result.epsilon << " "
                << result.frame_error_rate
                << "\\\\";
            log_line(log_file, msg.str());
        }
    }

    log_line(log_file, "% Done.");

    return 0;
}