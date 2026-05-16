/*
 * Copyright 2023 Sisi Miao, Communications Engineering Lab @ KIT
 *
 * SPDX-License-Identifier: MIT
 *
 * This file accompanies the paper
 *     S. Miao, A. Schnerring, H. Li and L. Schmalen,
 *     "Neural belief propagation decoding of quantum LDPC codes using overcomplete check matrices,"
 *     Proc. IEEE Inform. Theory Workshop (ITW), Saint-Malo, France, Apr. 2023, https://arxiv.org/abs/2212.10245
 */
#ifndef BPDECODING_STABILIZIERCODES_H
#define BPDECODING_STABILIZIERCODES_H
#include "fileReader.h"
//#include <experimental/filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <cmath>

class stabilizerCodes {
  public:
    stabilizerCodes(unsigned n, unsigned k, unsigned m,  const fileReader &fr);

    bool decode(unsigned int L, double epsilon,  int number_guess=1, double p = 0);
    bool binary_decode(unsigned int L, double epsilon, bool decimation = false);


    bool flooding_decode(unsigned int L, double epsilon);
    bool flooding_decode_joint(
    unsigned int L,
    double epsilon,
    double syndrome_p,
    const std::vector<std::vector<unsigned>> &Meta
    );

    bool genie_aided_BP(unsigned int L, double epsilon, int number_guess=1);
    std::vector<unsigned> flooding_decode_with_decimation(unsigned int L, double epsilon, std::vector<unsigned>&  error_last_bits, int number_guess=1);

    void flooding_decode_iteration(
    unsigned int decIter,
    double L0,
    std::vector<std::vector<double>> &mc2v,
    std::vector<std::vector<double>> &mv2c,
    std::vector<double> &Taux,
    std::vector<double> &Tauy,
    std::vector<double> &Tauz,
    std::vector<double> &phi_msg
    );


    bool multi_path_decoding(unsigned int L, double epsilon, int number_guess=1);

    std::vector<bool> check_success(const double *Taux, const double *Tauy, const double *Tauz);
    std::vector<unsigned> hard_decision(const double *Taux, const double *Tauy, const double *Tauz);
    bool check_success_true(std::vector<unsigned> ehat);

    static inline bool trace_inner_product(unsigned a, unsigned b);
    static inline double phi_func(double x);

    void add_error_given_epsilon(double epsilon);
    void add_one_type_of_error_given_epsilon(int error_type, double epsilon);
    void add_error_given_weight(unsigned w);

    void add_syndrome_error_given_p(double p);

    // void add_error_given_positions(int pos[], int error[], int size);

    void calculate_syndrome(); // also check if the error is all 0, if true, not decoding needed

    static double quantize_belief(double Taux, double Tauy, double Tauz);

    unsigned error_weight;
    unsigned max_iter;
    bool print_msg = false;


    unsigned N;
    unsigned K;
    unsigned M;
    unsigned G_rows;
    bool mTrained;
    unsigned trained_iter;

    std::vector<unsigned> error;
    std::vector<unsigned> error_hat;
    std::vector<unsigned> syn;
    std::vector<std::string> errorString;

    std::vector<unsigned> syndrome_error;
    std::vector<unsigned> correct_syndrome;
    unsigned maxDc;
    unsigned maxDv;
    std::vector<unsigned> dv;
    std::vector<unsigned> dc;
    std::vector<std::vector<unsigned>> Nv;
    std::vector<std::vector<unsigned>> Mc;
    std::vector<std::vector<unsigned>> checkVal;
    std::vector<std::vector<unsigned>> varVal;

    std::vector<std::vector<unsigned>> Nvk;
    std::vector<std::vector<unsigned>> Mck;

    std::vector<std::vector<unsigned>> G;
    std::vector<std::vector<unsigned>> Meta;

    int DecodingOption;
    int noise_model;

    static inline double clamp_msg(double x, double limit = 29.9)
    {
        if (x > limit) return limit;
        if (x < -limit) return -limit;
        return x;
    }

    static inline double binary_llr(double p)
    {//p: syndrome error rate, should hold p<0.5
        if (p <= 0.0) return 29.9;
        if (p >= 0.5) return 0.0;
        return std::log((1.0 - p) / p);
    }
};
#endif // BPDECODING_STABILIZIERCODES_H
