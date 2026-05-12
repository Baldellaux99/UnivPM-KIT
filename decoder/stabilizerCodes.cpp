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
#include "stabilizerCodes.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>

stabilizerCodes::stabilizerCodes(unsigned n, unsigned k, unsigned m,  const fileReader &fr) {
    N = n;
    K = k;
    M = m;
    G_rows = N + K;
    dc = fr.dc;
    dv = fr.dv;
    maxDc = fr.maxDc;
    maxDv = fr.maxDv;
    Nv = fr.Nv;
    Mc = fr.Mc;
    checkVal = fr.checkVal;
    varVal = fr.varVal;
    Nvk = fr.Nvk;
    Mck = fr.Mck;
    G = fr.G;
}

bool stabilizerCodes::decode(unsigned int L, double epsilon,int number_guess) {
    if (errorString.empty())
        return {true};

    if (record_histogram){for (unsigned i=0; i<L; i++){for (unsigned j=0; j<601; j++) {hist_vn_msg[i][j]=0;hist_cn_msg[i][j]=0;}}}

    calculate_syndrome();

    if (DecodingOption==0){
        return flooding_decode(L,epsilon);
    }
    else if (DecodingOption==1){
        return genie_aided_BP(L,epsilon, number_guess);
//        result = multi_path_decoding(L, epsilon, number_guess);

    }
    else if (DecodingOption==2){
        auto true_error = error;
        auto result = multi_path_decoding(L, epsilon, number_guess);
        return result;
    }
    else{
        std::runtime_error("decoding option not defined");
    }
    error_hat = std::vector<unsigned>(N, 0);
}



void stabilizerCodes::add_error_given_epsilon(double epsilon) {
    error.clear();
    errorString.clear();
    error_weight=0;
    std::uniform_real_distribution<double> dist(0, 1);

    auto res = std::random_device()();
    std::ranlux24 generator(res);
    std::uniform_real_distribution<double> distribution(0, 1);
    auto roll = [&distribution, &generator]() { return distribution(generator); };

    for (unsigned i = 0; i < N; i++) {
        double rndValue = roll();
        if (rndValue < epsilon / 3) {
            error.push_back(1);
            errorString.emplace_back("X" + std::to_string(i)); error_weight+=1;
        } else if (rndValue < epsilon * 2 / 3) {
            error.push_back(2);
            errorString.emplace_back("Z" + std::to_string(i)); error_weight+=1;
        } else if (rndValue < epsilon) {
            error.push_back(3);
            errorString.emplace_back("Y" + std::to_string(i)); error_weight+=1;
        } else {
            error.push_back(0);
        }
    }
}

void stabilizerCodes::add_one_type_of_error_given_epsilon(int error_type, double epsilon) {
    error.clear();
    errorString.clear();
    error_weight=0;
    std::uniform_real_distribution<double> dist(0, 1);

    auto res = std::random_device()();
    std::ranlux24 generator(res);
    std::uniform_real_distribution<double> distribution(0, 1);
    auto roll = [&distribution, &generator]() { return distribution(generator); };

    for (unsigned i = 0; i < N; i++) {
        double rndValue = roll();
        if (rndValue < epsilon) {
            error.push_back(error_type);
            errorString.emplace_back(std::to_string(i)); error_weight+=1;
        }  else {
            error.push_back(0);
        }
    }
}

void stabilizerCodes::add_error_given_weight(unsigned w) {
    error.clear();
    errorString.clear();
    error_weight = w;
    //chose random position
    std::random_device rd;
    std::mt19937 generator_pos(rd());

    std::vector<unsigned> nums; for (int i = 0; i < N; ++i) {nums.push_back(i);error.push_back(0);}
    unsigned seed = std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count();
    shuffle (nums.begin(), nums.end(), std::default_random_engine(seed));

    std::vector<unsigned> pos;
    for (int i = 0; i < w; ++i) {
        pos.push_back(nums[i]);
    }
    std::uniform_real_distribution<double> dist(0, 1);
    auto res = std::random_device()();
    std::ranlux24 generator(res);
    std::uniform_real_distribution<double> distribution(0, 1);
    auto roll = [&distribution, &generator]() { return distribution(generator); };


    for (unsigned i = 0; i < w; i++) {
        double rndValue = roll();
        if (rndValue < 1.0 / 3.0) {
            error[pos[i]]=1;
            errorString.emplace_back("X" + std::to_string(pos[i]));
        } else if (rndValue <  2.0 / 3.0) {
            error[pos[i]]=2;
            errorString.emplace_back("Z" + std::to_string(pos[i]));
        } else {
            error[pos[i]]=3;
            errorString.emplace_back("Y" + std::to_string(pos[i]));
        }
    }
}

// void stabilizerCodes::add_error_given_positions(int *pos, int *error, int size) {}

double stabilizerCodes::quantize_belief(double Tau, double Tau1, double Tau2) {
    double nom = log1p(exp(-1.0 * Tau));
    double denom = std::max(-1.0 * Tau1, -1.0 * Tau2) + log1p(exp(-1.0 * fabs((Tau1 - Tau2))));
    double ret_val = nom - denom;
    if (std::isnan(ret_val)) {
        throw std::runtime_error("quantize_belief: Log difference is NaN");
    }
    return ret_val;
}
//return true if anticommute
inline bool stabilizerCodes::trace_inner_product(unsigned int a, unsigned int b) {
    return !(a == 0 || b == 0 || a == b);
}

inline double stabilizerCodes::phi_func(double x) {
    return -1*log(tanh(x/2));
}


bool stabilizerCodes::flooding_decode(unsigned int L, double epsilon) {
    // Variable num_elements_in_H never used
    //
    // double num_elements_in_H = 0;
    // for (unsigned i = 0; i < N; i++) {
    //     num_elements_in_H += dv[i];
    // }
    std::vector<bool> success(2, false);
    double L0 = log(3.0 * (1 - epsilon) / epsilon);

    double lambda0 = log((1 + exp(-L0)) / (2 * exp(-L0)));
    //
    // Allocate memory for messages
    double **mc2v, **mv2c;
    mc2v = (double **)malloc(M * sizeof(double *));
    mv2c = (double **)malloc(N * sizeof(double *));
    double *Taux, *Tauz, *Tauy;
    Taux = (double *)malloc(N * sizeof(double *));
    Tauz = (double *)malloc(N * sizeof(double *));
    Tauy = (double *)malloc(N * sizeof(double *));
    double *phi_msg;
    phi_msg = (double *)malloc(maxDc * sizeof(double *));
    for (unsigned i = 0; i < M; i++) {
        mc2v[i] = (double *)malloc(dc[i] * sizeof(double *));
        for (unsigned j = 0; j < dc[i]; j++)
            mc2v[i][j] = 0;
    }
    for (unsigned i = 0; i < N; i++) {
        mv2c[i] = (double *)malloc(dv[i] * sizeof(double *));
        for (unsigned j = 0; j < dv[i]; j++) {
            mv2c[i][j] = lambda0;
            if (record_histogram){
                double flip_sign = trace_inner_product(error[i],varVal[i][j])?-1:1;
                double interval = (flip_sign*mv2c[i][j]+30);
                hist_vn_msg[0][(int)(interval*10)]+=1;
            }
        }
    }

    if (print_msg) {
        for (unsigned r = 0; r < 1; r++) {
            for (unsigned c = 0; c < dv[r]; c++) {
                std::cout << mv2c[r][c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // start decoding
    for (unsigned decIter = 0; decIter < L; decIter++) {
        // CN update, for i-th check node, calculate mc2v
        // Variable sum_cn_msg never used
        // double sum_cn_msg = 0;
        for (unsigned i = 0; i < M; i++) {
            double phi_sum = 0;
            double sign_prod = (syn[i] == 0 ? 1.0 : -1.0);

            // Sum-Product
            for (unsigned j = 0; j < dc[i]; j++) {

                if (mv2c[Mc[i][j]][Mck[i][j]] != 0.0)
                    phi_msg[j] = -1.0 * log(tanh(fabs(mv2c[Mc[i][j]][Mck[i][j]]) / 2.0));
                else
                    phi_msg[j] = 60;
                phi_sum += phi_msg[j];
                sign_prod *= (mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);
            }

            for (unsigned j = 0; j < dc[i]; j++) {
                double phi_extrinsic_phi_sum = phi_sum - phi_msg[j];
                double phi_phi_sum = 60;
                if (phi_extrinsic_phi_sum != 0)
                    phi_phi_sum = -1.0 * log(tanh(phi_extrinsic_phi_sum / 2.0));
                mc2v[i][j] = phi_phi_sum * sign_prod * (mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);

                if (record_histogram){
                    double flip_sign = trace_inner_product(error[Mck[i][j]],checkVal[i][j])?-1:1;
                    double interval = (flip_sign*mc2v[i][j]+30);
                    hist_cn_msg[decIter][(int)(interval*10)]+=1;
                }
                // sum_cn_msg += fabs(mc2v[i][j]);
                if (std::isnan(mc2v[i][j])) {
                    std::cout<<"iter: "<<decIter<<std::endl;
                    throw std::runtime_error("flooding_decode: mc2v[i][j] is NaN");
                }
                if (std::isinf(mc2v[i][j])) {
                    std::cout<<"iter: "<<decIter<<std::endl;
                    throw std::runtime_error("flooding_decode: mc2v[i][j] is infinity");
                }
            }
        }

        // unused:
        // double ave_cn_msg = sum_cn_msg / num_elements_in_H;
        if (print_msg) {
            std::cout << "CN messages" << std::endl;
            for (unsigned r = 0; r < 1; r++) {
                for (unsigned c = 0; c < dc[r]; c++) {
                    std::cout << mc2v[r][c] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        // VN update
        for (unsigned Vidx = 0; Vidx < N; Vidx++) {
            Taux[Vidx] = L0;
            Tauz[Vidx] = L0;
            Tauy[Vidx] = L0;
            double Tauxi;
            double Tauyi;
            double Tauzi;

            // jj, index of the neighboring CNs of the VN, sum up the CN messages
            for (unsigned jj = 0; jj < dv[Vidx]; jj++) {
                if (varVal[Vidx][jj] == 1) {
                    Tauz[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauy[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else if (varVal[Vidx][jj] == 2) {
                    Taux[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauy[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else if (varVal[Vidx][jj] == 3) {
                    Taux[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauz[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else
                    throw std::invalid_argument("something is wrong");
            }

            for (unsigned jj = 0; jj < dv[Vidx]; jj++) {
                double temp;
                if (varVal[Vidx][jj] == 1) {
                    Tauxi = Taux[Vidx];
                    Tauzi = Tauz[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauyi = Tauy[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    temp = quantize_belief(Tauxi, Tauyi, Tauzi);
                } else if (varVal[Vidx][jj] == 2) {
                    Tauxi = Taux[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauzi = Tauz[Vidx];
                    Tauyi = Tauy[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    temp = quantize_belief(Tauzi, Tauyi, Tauxi);
                } else if (varVal[Vidx][jj] == 3) {
                    Tauxi = Taux[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauzi = Tauz[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauyi = Tauy[Vidx];
                    temp = quantize_belief(Tauyi, Tauxi, Tauzi);
                } else
                    throw std::invalid_argument("something is wrong");
                double limit = 29.9;
                if (temp > limit)
                    mv2c[Vidx][jj] = limit;
                else if (temp < -limit)
                    mv2c[Vidx][jj] = -limit;
                else
                    mv2c[Vidx][jj] = temp;

                if (std::isnan(mv2c[Vidx][jj])) {
                    std::cout<<"iter: "<<decIter<<std::endl;
                    throw std::runtime_error("flooding_decode: mv2c[Vidx][jj] is NaN");
                }
                if (std::isinf(mv2c[Vidx][jj])) {
                    std::cout<<"iter: "<<decIter<<std::endl;
                    throw std::runtime_error("flooding_decode: mv2c[Vidx][jj] is infinity");

                }

                if (record_histogram){
                    double flip_sign = trace_inner_product(error[Vidx],varVal[Vidx][jj])?-1:1;
                    double interval = (flip_sign * mv2c[Vidx][jj]+30);
                    hist_vn_msg[decIter+1][(int)(interval*10)]+=1;
                }
            }
        }
        if (print_msg) {
            std::cout << "Taux, Tauy, Tauz" << std::endl;
            for (unsigned i = 0; i < N; i++)
                std::cout << Taux[i] << " ";
            std::cout << std::endl;
            for (unsigned i = 0; i < N; i++)
                std::cout << Tauy[i] << " ";
            std::cout << std::endl;
            for (unsigned i = 0; i < N; i++)
                std::cout << Tauz[i] << " ";
            std::cout << std::endl;
            std::cout << "VN messages" << std::endl;
            for (unsigned r = 0; r < 1; r++) {
                for (unsigned c = 0; c < dv[r]; c++) {
                    std::cout << mv2c[r][c] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        success = check_success(Taux, Tauy, Tauz);
        if (success[0]){max_iter = decIter; break;}

    }
    if (!success[0]) max_iter = L;
    free(Taux);
    free(Tauy);
    free(Tauz);
    free(phi_msg);
    for (unsigned i = 0; i < M; i++) {
        free(mc2v[i]);
    }
    for (unsigned i = 0; i < N; i++) {
        free(mv2c[i]);
    }
    free(mc2v);
    free(mv2c);

    return success[1];
}

void stabilizerCodes::calculate_syndrome() {
    for (unsigned i = 0; i < M; i++) {
        bool check = false;
        for (unsigned j = 0; j < dc[i]; j++) {
            check = trace_inner_product(error[Mc[i][j]], checkVal[i][j]) ? !check : check;
        }
        syn.push_back(check % 2);
    }
}

std::vector<bool> stabilizerCodes::check_success(const double *Taux, const double *Tauy, const double *Tauz) {
    std::vector<bool> success(4, false);
    error_hat = std::vector<unsigned>(N, 0);
    for (unsigned i = 0; i < N; i++) {
        if (Taux[i] > 0 && Tauy[i] > 0 && Tauz[i] > 0) {
            error_hat[i] = 0;
        } else if (Taux[i] < Tauy[i] && Taux[i] < Tauz[i]) {
            error_hat[i] = 1;
        } else if (Tauz[i] < Taux[i] && Tauz[i] < Tauy[i]) {
            error_hat[i] = 2;
        } else {
            error_hat[i] = 3;
        }
    }
    unsigned error_hat_weight = 0;
    for (unsigned i=0; i < N; i++){
        if (error[i]!=error_hat[i]) {success[2] = true; break;}
    }
    for (unsigned i=0; i < N; i++){
        if (error_hat[i]!=0) {error_hat_weight+=1;}
    }
    success[3] = (error_hat_weight>error_weight);
    for (unsigned i = 0; i < M; i++) {
        // unsigned check = 0;
        bool check = false;
        for (unsigned j = 0; j < dc[i]; j++) {
            check = trace_inner_product(error_hat[Mc[i][j]], checkVal[i][j]) ? !check : check;
        }
        if ((check) != syn[i]) {
            return success;
        }
    }
    success[0] = true;

    for (unsigned i = 0; i < G_rows; i++) {
        bool check = false;
        for (unsigned j = 0; j < N; j++) {
            check = trace_inner_product(error[j], G[i][j]) ? !check : check;
            check = trace_inner_product(error_hat[j], G[i][j]) ? !check : check;
        }
        if (check) {
            return success;
        }
    }
    success[1] = true;
    return success;
}

std::vector<unsigned> stabilizerCodes::hard_decision(const double *Taux, const double *Tauy, const double *Tauz) {
    std::vector<unsigned> e_hat = std::vector<unsigned>(N, 0);
    for (unsigned i = 0; i < N; i++) {
        if (Taux[i] > 0 && Tauy[i] > 0 && Tauz[i] > 0) {e_hat[i] = 0;}
        else if (Taux[i] < Tauy[i] && Taux[i] < Tauz[i]) {e_hat[i] = 1;}
        else if (Tauz[i] < Taux[i] && Tauz[i] < Tauy[i]) {e_hat[i] = 2;}
        else {e_hat[i] = 3;}
    }
    for (unsigned i = 0; i < M; i++) {
        // unsigned check = 0;
        bool check = false;
        for (unsigned j = 0; j < dc[i]; j++) {check = trace_inner_product(e_hat[Mc[i][j]], checkVal[i][j]) ? !check : check;}
        if ((check) != syn[i]) {
            e_hat.push_back(0);//last bit of e_hat denotes if syndrome is matched or not
            return e_hat;
        }
    }
    e_hat.push_back(1);
    return e_hat;
}

std::vector<unsigned> stabilizerCodes::flooding_decode_with_decimation(unsigned int L, double epsilon, std::vector<unsigned> &error_last_bits, int number_guess) {
    std::vector<bool> success(2, false);
    double L0 = log(3.0 * (1 - epsilon) / epsilon);
    //unsigned decimated_bit = N-1;
    std::vector<unsigned> decimated_bits (number_guess,0);
    for (int i=0; i<number_guess; i++) decimated_bits[i] = N-1-i;
    std::vector<unsigned> e_hat;
    double lambda0 = log((1 + exp(-L0)) / (2 * exp(-L0)));
    //
    // Allocate memory for messages
    double **mc2v, **mv2c;
    mc2v = (double **)malloc(M * sizeof(double *));mv2c = (double **)malloc(N * sizeof(double *));
    double *Taux, *Tauz, *Tauy;
    Taux = (double *)malloc(N * sizeof(double *));Tauz = (double *)malloc(N * sizeof(double *));Tauy = (double *)malloc(N * sizeof(double *));
    double *phi_msg;
    phi_msg = (double *)malloc(maxDc * sizeof(double *));
    for (unsigned i = 0; i < M; i++) {mc2v[i] = (double *)malloc(dc[i] * sizeof(double *));for (unsigned j = 0; j < dc[i]; j++)mc2v[i][j] = 0;}
    for (unsigned i = 0; i < N; i++) {mv2c[i] = (double *)malloc(dv[i] * sizeof(double *));for (unsigned j = 0; j < dv[i]; j++) {mv2c[i][j] = lambda0;}}
    //speical treatment for the last bit
//    mv2c[N-1] = (double *)malloc(dv[N-1] * sizeof(double *));
    for (int i=0; i<number_guess; i++){
        unsigned decimated_bit = decimated_bits[i];
        for (unsigned j = 0; j < dv[decimated_bit]; j++) {mv2c[decimated_bit][j] = (trace_inner_product(error_last_bits[i], varVal[decimated_bit][j])) ? 1e10 : -1e10;}
    }


    if (print_msg) {
        for (unsigned r = 0; r < 1; r++) {
            for (unsigned c = 0; c < dv[r]; c++) {
                std::cout << mv2c[r][c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // start decoding
    for (unsigned decIter = 0; decIter < L; decIter++) {
        // CN update, for i-th check node, calculate mc2v
        for (unsigned i = 0; i < M; i++) {
            double phi_sum = 0;double sign_prod = (syn[i] == 0 ? 1.0 : -1.0);

            // Sum-Product
            for (unsigned j = 0; j < dc[i]; j++) {
                if (mv2c[Mc[i][j]][Mck[i][j]] != 0.0) phi_msg[j] = -1.0 * log(tanh(fabs(mv2c[Mc[i][j]][Mck[i][j]]) / 2.0));
                else phi_msg[j] = 60;
                phi_sum += phi_msg[j];
                sign_prod *= (mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);
            }

            for (unsigned j = 0; j < dc[i]; j++) {
                double phi_extrinsic_phi_sum = phi_sum - phi_msg[j];
                double phi_phi_sum = 60;
                if (phi_extrinsic_phi_sum != 0)phi_phi_sum = -1.0 * log(tanh(phi_extrinsic_phi_sum / 2.0));
                mc2v[i][j] = phi_phi_sum * sign_prod * (mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);

                // sum_cn_msg += fabs(mc2v[i][j]);
                if (std::isnan(mc2v[i][j])) {std::cout<<"iter: "<<decIter<<std::endl;throw std::runtime_error("flooding_decode: mc2v[i][j] is NaN");}
                if (std::isinf(mc2v[i][j])) {std::cout<<"iter: "<<decIter<<std::endl;throw std::runtime_error("flooding_decode: mc2v[i][j] is infinity");}
            }
        }
        if (print_msg) {
            std::cout<<"iter "<<decIter<<"\n";
            std::cout << "CN messages" << std::endl;
            for (unsigned r = 0; r < 1; r++) {for (unsigned c = 0; c < dc[r]; c++) {std::cout << mc2v[r][c] << " ";} std::cout << std::endl;}std::cout << std::endl;}



        // VN update for all bits
        for (unsigned Vidx = 0; Vidx < N - number_guess; Vidx++) {
            Taux[Vidx] = L0;
            Tauz[Vidx] = L0;
            Tauy[Vidx] = L0;
        }

        //VN update for decimated bits
        for (int ng=0; ng<number_guess; ng++) {
            unsigned decimated_bit = decimated_bits[ng];
            unsigned error_last_bit = error_last_bits[ng];
            if (error_last_bit == 0) {
                Taux[decimated_bit] = 1e10;
                Tauz[decimated_bit] = 1e10;
                Tauy[decimated_bit] = 1e10;
            }
            else if (error_last_bit == 1) {
                Taux[decimated_bit] = -1e10;
                Tauz[decimated_bit] = 1e10;
                Tauy[decimated_bit] = 1e10;
            }
            else if (error_last_bit == 2) {
                Taux[decimated_bit] = 1e10;
                Tauz[decimated_bit] = -1e10;
                Tauy[decimated_bit] = 1e10;
            }
            else {
                Taux[decimated_bit] = 1e10;
                Tauz[decimated_bit] = 1e10;
                Tauy[decimated_bit] = -1e10;
            }

        }

        for (unsigned Vidx = 0; Vidx < N; Vidx++) {
            double Tauxi;double Tauyi;double Tauzi;

            // jj, index of the neighboring CNs of the VN, sum up the CN messages
            for (unsigned jj = 0; jj < dv[Vidx]; jj++) {
                if (varVal[Vidx][jj] == 1) {
                    Tauz[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];Tauy[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else if (varVal[Vidx][jj] == 2) {
                    Taux[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];Tauy[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else if (varVal[Vidx][jj] == 3) {
                    Taux[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];Tauz[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else
                    throw std::invalid_argument("something is wrong");
            }

            for (unsigned jj = 0; jj < dv[Vidx]; jj++) {
                double temp;
                if (varVal[Vidx][jj] == 1) {
                    Tauxi = Taux[Vidx];Tauzi = Tauz[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];Tauyi = Tauy[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    temp = quantize_belief(Tauxi, Tauyi, Tauzi);
                } else if (varVal[Vidx][jj] == 2) {
                    Tauxi = Taux[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];Tauzi = Tauz[Vidx];Tauyi = Tauy[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    temp = quantize_belief(Tauzi, Tauyi, Tauxi);
                } else if (varVal[Vidx][jj] == 3) {
                    Tauxi = Taux[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];Tauzi = Tauz[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];Tauyi = Tauy[Vidx];
                    temp = quantize_belief(Tauyi, Tauxi, Tauzi);
                } else
                    throw std::invalid_argument("something is wrong");
                double limit = 29.9;
                if (temp > limit)mv2c[Vidx][jj] = limit;else if (temp < -limit)mv2c[Vidx][jj] = -limit;else mv2c[Vidx][jj] = temp;

                if (std::isnan(mv2c[Vidx][jj])) {std::cout<<"iter: "<<decIter<<std::endl;throw std::runtime_error("flooding_decode: mv2c[Vidx][jj] is NaN");}
                if (std::isinf(mv2c[Vidx][jj])) {std::cout<<"iter: "<<decIter<<std::endl;throw std::runtime_error("flooding_decode: mv2c[Vidx][jj] is infinity");}

            }
        }


//        for (unsigned j = 0; j < dv[decimated_bit]; j++) {mv2c[decimated_bit][j] = (trace_inner_product(error_last_bit, varVal[decimated_bit][j])) ? 1e10 : -1e10;}
        if (print_msg) {
            std::cout << "Taux, Tauy, Tauz" << std::endl;
            for (unsigned i = 0; i < N; i++)std::cout << Taux[i] << " ";std::cout << std::endl;
            for (unsigned i = 0; i < N; i++)std::cout << Tauy[i] << " ";std::cout << std::endl;
            for (unsigned i = 0; i < N; i++)std::cout << Tauz[i] << " ";std::cout << std::endl;
            std::cout << "VN messages" << std::endl;
            for (unsigned r = 0; r < N; r++) {for (unsigned c = 0; c < dv[r]; c++) {std::cout << mv2c[r][c] << " ";}std::cout << std::endl;}std::cout << std::endl;
        }
        e_hat = hard_decision(Taux, Tauy, Tauz);
        if (e_hat[N]==1){
//            e_hat[N] = decIter;
            break;}

    }

    free(Taux);free(Tauy);free(Tauz);free(phi_msg);
    for (unsigned i = 0; i < M; i++) {free(mc2v[i]);}for (unsigned i = 0; i < N; i++) {free(mv2c[i]);}
    free(mc2v);free(mv2c);
    return e_hat;
}

bool stabilizerCodes::multi_path_decoding(unsigned int L, double epsilon, int number_guess) {
    auto true_error = error;
    std::vector<std::vector<unsigned >> error_canidates;
    std::vector<unsigned> candidate_iter;
    unsigned num_canidates = 0;

    for (int i=0; i<pow(4,number_guess); i++){
        std::vector<unsigned> error_last_bits(number_guess, 0);
        unsigned dec = i;
        for (int j=0; j<number_guess; j++){
            error_last_bits[j] = dec % 4;
            dec /= 4;
        }

        auto e_hat = flooding_decode_with_decimation(L, epsilon, error_last_bits, number_guess);
        if (e_hat[N]>0) {candidate_iter.push_back(e_hat[N]); e_hat.pop_back();error_canidates.push_back(e_hat);num_canidates++;}
    }
    if (num_canidates==0)
        return false;
    if (num_canidates==1){
        auto result = check_success_true(error_canidates[0]);
        return result;
    }
    unsigned min_idx = 0;
    //choosing based on error weight
    unsigned min_weight_sum=N+1;
    for (int i=0; i<num_canidates; i++){
        unsigned sum=0; for (int j=0; j<N; j++){if (error_canidates[i][j]!=0) sum++;}
        if (sum<min_weight_sum){
            min_weight_sum = sum;
            min_idx = i;}
    }
    //choosing based on maxIter for low decoding latency -> not good
//    unsigned min_iter = L+1;
//    for (int i=0; i<num_canidates; i++){
//        if (candidate_iter[i]<min_iter){min_iter = i; min_iter = candidate_iter[i];}
//    }
    error_hat = error_canidates[min_idx];
    auto result = check_success_true(error_canidates[min_idx]);
//    if (min_weight_sum<error_weight && !result){
//        std::cout<<"lower weight casued error"<<std::endl;
//    }
    return result;
}

bool stabilizerCodes::check_success_true(std::vector<unsigned int> ehat) {
    for (unsigned i = 0; i < G_rows; i++) {
        bool check = false;
        for (unsigned j = 0; j < N; j++) {
            check = trace_inner_product(error[j], G[i][j]) ? !check : check;
            check = trace_inner_product(ehat[j], G[i][j]) ? !check : check;
        }
        if (check) {
            return false;
        }
    }
    return true;
}

bool stabilizerCodes::genie_aided_BP(unsigned int L, double epsilon, int number_guess) {
    std::vector<unsigned> error_last_bits(number_guess, 0);
    for (int i=0; i<number_guess; i++){
        error_last_bits[i] = error[N-i - 1] ;
    }
    auto error_canidate = flooding_decode_with_decimation(L,epsilon,error_last_bits, number_guess);
    error_hat = error_canidate;
    return check_success_true(error_canidate);
}

bool stabilizerCodes::binary_decode(unsigned int L, double epsilon, bool decimation) {
    //initialization, locating memory
    if (errorString.empty()) return {true};

    calculate_syndrome();

    std::vector<bool> success(2, false);
    double L0 = log((1 - 2*epsilon/3) / (2*epsilon/3));


    double **mc2v, **mv2c;
    mc2v = (double **)malloc(M * sizeof(double *));
    mv2c = (double **)malloc(N * sizeof(double *));

    for (unsigned i = 0; i < M; i++) {
        mc2v[i] = (double *)malloc(dc[i] * sizeof(double *));
        for (unsigned j = 0; j < dc[i]; j++) mc2v[i][j] = 0;
    }
    for (unsigned i = 0; i < N; i++) {
        mv2c[i] = (double *)malloc(dv[i] * sizeof(double *));
        for (unsigned j = 0; j < dv[i]; j++) {
            if (Nv[i][j]<M/2) mv2c[i][j] = L0;
            else mv2c[i][j] = 0;
        }
    }
    bool last_bit_no_error = !(error[N-1]==2 || error[N-1]==3);
    if (decimation){
        for (unsigned j = 0; j < dv[N-1]; j++)
        {if (Nv[N-1][j]<M/2)
                mv2c[N-1][j] = last_bit_no_error ? 1e10 : -1e10;
        }
    }

    //decoding z errors using Hx======================================================================================
    std::vector<unsigned> error_hat_z = std::vector<unsigned>(N, 0);
    bool converged = true;
    for (unsigned decIter = 0; decIter < L; decIter++) {

        for (int i=0; i<M/2; i++){
            double sign=(syn[i] == 0 ? 1.0 : -1.0);
            double ampl = 0;
            for (unsigned j = 0; j < dc[i]; j++) {
                sign *=  (mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);
                ampl += phi_func(fabs(mv2c[Mc[i][j]][Mck[i][j]]));
            }
            for (unsigned j = 0; j < dc[i]; j++) {
                double phi_extrinsic_sum = phi_func(ampl - phi_func(fabs(mv2c[Mc[i][j]][Mck[i][j]])));
                double extrinsic_sign = sign*(mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);
                mc2v[i][j] =  extrinsic_sign * phi_extrinsic_sum;
            }
        }
//        std::cout << "CN messages" << std::endl;
//        for (unsigned r = 0; r < M; r++) {for (unsigned c = 0; c < dc[r]; c++) {std::cout << mc2v[r][c] << " ";}std::cout << std::endl;}std::cout << std::endl;
//       std::cout << "intrinsic sum" << std::endl;
        for (int i=0; i<N; i++){
            double intrinsic = L0;
            if (i==N-1&& decimation && last_bit_no_error) intrinsic = 1e10;
            if (i==N-1&& decimation && !last_bit_no_error) intrinsic = -1e10;
            for (unsigned j=0; j<dv[i]; j++){
                if (Nv[i][j]<M/2) intrinsic += mc2v[Nv[i][j]][Nvk[i][j]];
            }
            if (intrinsic>0) error_hat_z[i] = 0; else error_hat_z[i]=2;
//            std::cout<<intrinsic<<" ";
            for (unsigned j=0; j<dv[i]; j++){
                if (Nv[i][j]<M/2) {
                    double tmp = intrinsic - mc2v[Nv[i][j]][Nvk[i][j]];
                    if (tmp < -30) tmp = -30;
                    if (tmp > 30) tmp = 30;
                    mv2c[i][j] = tmp;
                }
            }
        }

        converged = true;
        for (unsigned i=0; i<M/2; i++){
            bool check = false;
            for (unsigned j = 0; j < dc[i]; j++) {
                check = trace_inner_product(error_hat_z[Mc[i][j]], checkVal[i][j]) ? !check : check;
            }
            if ((check) != syn[i]) {
                converged = false;
                break;
            }
        }
        if (converged)
            break;
    }
    if (!converged) return false;


    //decoding X errors using Hz=======================================================================================

    for (unsigned i = 0; i < M; i++) {for (unsigned j = 0; j < dc[i]; j++)mc2v[i][j] = 0;}
    for (unsigned i = 0; i < N; i++) {for (unsigned j = 0; j < dv[i]; j++) {mv2c[i][j] = L0;}}
    last_bit_no_error = !(error[N-1]==1 || error[N-1]==3);
    if (decimation){
        for (unsigned j = 0; j < dv[N-1]; j++)
        {if (Nv[N-1][j]>=M/2)
                mv2c[N-1][j] = last_bit_no_error ? 1e10 : -1e10;
        }
    }
    error_hat = std::vector<unsigned>(N, 0);
    for (unsigned decIter = 0; decIter < L; decIter++) {
        for (int i=M/2+1; i<M; i++){
            double sign=(syn[i] == 0 ? 1.0 : -1.0);
            double ampl = 0;
            for (unsigned j = 0; j < dc[i]; j++) {
                sign *=  (mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);
                ampl += phi_func(fabs(mv2c[Mc[i][j]][Mck[i][j]]));
            }
            for (unsigned j = 0; j < dc[i]; j++) {
                double extrinsic_sum = ampl - phi_func(fabs(mv2c[Mc[i][j]][Mck[i][j]]) );
                double phi_extrinsic_sum = phi_func(extrinsic_sum);
                double extrinsic_sign = sign*(mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);
                mc2v[i][j] =  extrinsic_sign * phi_extrinsic_sum;
            }
        }
//        std::cout << "CN messages" << std::endl;
//        for (unsigned r = 0; r < M; r++) {for (unsigned c = 0; c < dc[r]; c++) {std::cout << mc2v[r][c] << " ";}std::cout << std::endl;}std::cout << std::endl;

        for (int i=0; i<N; i++){
            double intrinsic = L0;
            if (i==N-1&& decimation && last_bit_no_error) intrinsic = 1e10;
            if (i==N-1&& decimation && !last_bit_no_error) intrinsic = -1e10;
            for (unsigned j=0; j<dv[i]; j++){
                if (Nv[i][j]>=M/2) intrinsic += mc2v[Nv[i][j]][Nvk[i][j]];
            }
            if (intrinsic<0) {error_hat[i] = 1;} else {error_hat[i]=0;}
            for (unsigned j=0; j<dv[i]; j++){
                if (Nv[i][j]>=M/2){
                double tmp = intrinsic - mc2v[Nv[i][j]][Nvk[i][j]];
                if (tmp<-30) tmp = -30; if (tmp>30) tmp = 30;
                mv2c[i][j] = tmp;
                }
            }
        }


        converged = true;
        for (unsigned i=M/2+1; i<M; i++){
            bool check = false;
            for (unsigned j = 0; j < dc[i]; j++) {
                check = trace_inner_product(error_hat[Mc[i][j]], checkVal[i][j]) ? !check : check;
            }
            if ((check) != syn[i]) {converged = false; break;}
        }

        if (converged)
            break;
    }
    if (!converged) return false;

    for (unsigned i = 0; i < M; i++) {free(mc2v[i]);}for (unsigned i = 0; i < N; i++) {free(mv2c[i]);}free(mc2v);free(mv2c);

    for (unsigned i=0; i<N; i++) error_hat[i]+=error_hat_z[i];
    return check_success_true(error_hat);
}


