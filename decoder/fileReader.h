//
// Created by sisi on 4/20/23.
//

#ifndef NBP_JUPYTER_FILEREADER_H
#define NBP_JUPYTER_FILEREADER_H

#endif // NBP_JUPYTER_FILEREADER_H
//#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

class fileReader {
  public:
    fileReader(unsigned n, unsigned k, unsigned m);
    void read_H();
    void read_G();

    bool check_symplectic();
    static inline bool trace_inner_product(unsigned a, unsigned b);
    unsigned N;
    unsigned K;
    unsigned M;
    unsigned G_rows;

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

    std::vector<std::vector<std::vector<double>>> weights_cn;
    std::vector<std::vector<std::vector<double>>> weights_vn;
    std::vector<std::vector<double>> weights_llr;

    std::vector<std::vector<unsigned>> LUTx;
    std::vector<std::vector<unsigned>> LUTz;
};
