//
// Created by sisi on 4/20/23.
//
#include "fileReader.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

fileReader::fileReader(unsigned n, unsigned k, unsigned m) {
    N = n;
    K = k;
    M = m;
    G_rows = N + K;
    std::vector<std::vector<unsigned>> Meta;
    read_H();
    read_G();
    read_Meta();
}

void fileReader::read_H() {

    std::string filename = "./PCMsForPLmodel/" + std::to_string(N) + "_" + std::to_string(K)+ "/"
            +std::to_string(N) + "_" + std::to_string(K) +"_"+std::to_string(M)+"_H" + ".txt";
    // Nvk same shape and Nv, but its value k means the i-th VN being the k-th neigbor of j-th CN
    // Mck same shape and Nv, but its value k means the i-th CN being the k-th neigbor of j-th VN
    std::vector<std::vector<unsigned>> checkValues;
    std::vector<std::vector<unsigned>> VariableValues;

    std::string line;
    std::ifstream matrix_file;
    matrix_file.open(filename);
    if (matrix_file.is_open()) {
        // first line, n k
        getline(matrix_file, line, ' ');
        unsigned n = std::stoul(line);
        getline(matrix_file, line);
        unsigned m = std::stoul(line);

        if (n != N) {
            throw std::runtime_error("read_H: file-specified N not as expected");
        }
        if (m != M) {
            throw std::runtime_error("read_H: file-specified M not as expected");
        }
        // second line max dv and dc
        getline(matrix_file, line, ' ');
        maxDv = std::stoul(line);
        getline(matrix_file, line);
        maxDc = std::stoul(line);

        // third line, dv degrees
        for (unsigned i = 0; i < n - 1; i++) {
            getline(matrix_file, line, ' ');
            dv.push_back(std::stoul(line));
            Nvk.emplace_back();
        }
        Nvk.emplace_back();
        getline(matrix_file, line);
        dv.push_back(std::stoul(line));

        // forth line, dc degrees
        for (unsigned i = 0; i < m - 1; i++) {
            getline(matrix_file, line, ' ');
            dc.push_back(std::stoul(line));
            Mck.emplace_back();
        }
        Mck.emplace_back();
        getline(matrix_file, line);
        dc.push_back(std::stoul(line));

        // fifth to n+5-th line, dv neighbors
        for (unsigned i = 0; i < n; i++) {
            Nv.emplace_back(dv[i], 0);
            for (unsigned j = 0; j < dv[i] - 1; j++) {
                getline(matrix_file, line, ' ');
                Nv[i][j] = std::stoul(line) - 1;
                Mck[Nv[i][j]].push_back(j);
            }
            getline(matrix_file, line);
            Nv[i][dv[i] - 1] = std::stoul(line) - 1;
            Mck[Nv[i][dv[i] - 1]].push_back(dv[i] - 1);
        }

        // n+6-th to n+6+m-th line, dc neighbors
        for (unsigned i = 0; i < m; i++) {
            Mc.emplace_back(dc[i], 0);
            for (unsigned j = 0; j < dc[i] - 1; j++) {
                getline(matrix_file, line, ' ');
                Mc[i][j] = std::stoul(line) - 1;
                Nvk[Mc[i][j]].push_back(j);
            }
            getline(matrix_file, line);
            Mc[i][dc[i] - 1] = std::stoul(line) - 1;
            Nvk[Mc[i][dc[i] - 1]].push_back(dc[i] - 1);
        }
        // n+6+m+1-th to n+6+m+n-th line, value of each row
        for (unsigned i = 0; i < m; i++) {
            checkValues.emplace_back(dc[i], 0);
            for (unsigned j = 0; j < dc[i] - 1; j++) {
                getline(matrix_file, line, ' ');
                checkValues[i][j] = std::stoul(line);
            }
            getline(matrix_file, line);
            checkValues[i][dc[i] - 1] = std::stoul(line);
        }
        // last lines, value of each column
        for (unsigned i = 0; i < n; i++) {
            VariableValues.emplace_back(dv[i], 0);
            for (unsigned j = 0; j < dv[i] - 1; j++) {
                getline(matrix_file, line, ' ');
                VariableValues[i][j] = std::stoul(line);
            }
            getline(matrix_file, line);
            VariableValues[i][dv[i] - 1] = std::stoul(line);
        }
        matrix_file.close();
    } else
        std::cout << "Unable to open file" << std::endl;
    checkVal = checkValues;
    varVal = VariableValues;
}
void fileReader::read_G() {
    std::string filename = "./PCMsForPLmodel/" + std::to_string(N) + "_" + std::to_string(K) + "/" +
                           std::to_string(N) + "_" + std::to_string(K) + "_G.txt";
    std::string line;
    std::ifstream matrix_file;
    matrix_file.open(filename);
    if (matrix_file.is_open()) {
        for (unsigned i = 0; i < G_rows; i++) {
            std::vector<unsigned> row(N, 0);
            for (unsigned j = 0; j < N - 1; j++) {
                getline(matrix_file, line, ' ');
                row[j] = std::stoul(line);
            }
            getline(matrix_file, line);
            row[N - 1] = std::stoul(line);
            G.push_back(row);
        }
        matrix_file.close();
    } else
        std::cout << "Unable to open file" << std::endl;
}


void fileReader::read_Meta()
{
    std::string filename =
        "./PCMsForPLmodel/" + std::to_string(N) + "_" + std::to_string(K) + "/" +
        std::to_string(N) + "_" + std::to_string(K) + "_Mcss_meta.txt";

    std::ifstream matrix_file(filename);

    if (!matrix_file.is_open())
    {
        throw std::runtime_error("read_Meta: unable to open file " + filename);
    }

    unsigned n_file = 0;  // number of columns
    unsigned m_file = 0;  // number of rows

    matrix_file >> n_file >> m_file;

    // Mcss_meta acts on the full syndrome vector, so its number of columns
    // must equal the number of stabilizer checks M.
    if (n_file != M)
    {
        throw std::runtime_error(
            "read_Meta: file-specified number of columns does not match M"
        );
    }

    unsigned max_col_weight = 0;
    unsigned max_row_weight = 0;

    matrix_file >> max_col_weight >> max_row_weight;

    std::vector<unsigned> col_weight(n_file, 0);
    std::vector<unsigned> row_weight(m_file, 0);

    for (unsigned j = 0; j < n_file; ++j)
    {
        matrix_file >> col_weight[j];
    }

    for (unsigned i = 0; i < m_file; ++i)
    {
        matrix_file >> row_weight[i];
    }

    // ------------------------------------------------------------
    // Read column neighbor indices.
    // We do not need them for dense Meta, but we must consume them.
    // Indices in file are 1-based and padded with 0.
    // ------------------------------------------------------------

    for (unsigned j = 0; j < n_file; ++j)
    {
        for (unsigned t = 0; t < max_col_weight; ++t)
        {
            unsigned idx = 0;
            matrix_file >> idx;
        }
    }

    // ------------------------------------------------------------
    // Read row neighbor indices.
    // These are the syndrome-bit positions touched by each meta-check.
    // ------------------------------------------------------------

    std::vector<std::vector<unsigned>> row_indices(m_file);

    for (unsigned i = 0; i < m_file; ++i)
    {
        row_indices[i].reserve(row_weight[i]);

        for (unsigned t = 0; t < max_row_weight; ++t)
        {
            unsigned idx = 0;
            matrix_file >> idx;

            if (t < row_weight[i])
            {
                if (idx == 0 || idx > n_file)
                {
                    throw std::runtime_error("read_Meta: invalid row neighbor index");
                }

                row_indices[i].push_back(idx - 1);  // convert to 0-based
            }
        }
    }

    // ------------------------------------------------------------
    // Read row neighbor values and build dense Meta.
    // For Mcss_meta, values should be binary, usually all 1.
    // ------------------------------------------------------------

    Meta.clear();
    Meta.assign(m_file, std::vector<unsigned>(n_file, 0));

    for (unsigned i = 0; i < m_file; ++i)
    {
        for (unsigned t = 0; t < max_row_weight; ++t)
        {
            unsigned val = 0;
            matrix_file >> val;

            if (t < row_weight[i])
            {
                Meta[i][row_indices[i][t]] = val & 1U;
            }
        }
    }

    // ------------------------------------------------------------
    // Read column neighbor values.
    // We do not need them for dense Meta, but we must consume them.
    // ------------------------------------------------------------

    for (unsigned j = 0; j < n_file; ++j)
    {
        for (unsigned t = 0; t < max_col_weight; ++t)
        {
            unsigned val = 0;
            matrix_file >> val;
        }
    }

    if (!matrix_file.good() && !matrix_file.eof())
    {
        throw std::runtime_error("read_Meta: error while reading file");
    }

    matrix_file.close();

    std::cout << "Read Meta from: " << filename << std::endl;
    std::cout << "Meta shape: " << Meta.size() << " x "
              << (Meta.empty() ? 0 : Meta[0].size()) << std::endl;
}



bool fileReader::check_symplectic() {
    //    read_H();
    //    read_G();
    unsigned *vec1;
    vec1 = (unsigned *)malloc(N * sizeof(unsigned *));
    unsigned *vec2;
    vec2 = (unsigned *)malloc(N * sizeof(unsigned *));

    for (unsigned rowid = 0; rowid < M; rowid++) {
        for (unsigned iii = 0; iii < N; iii++)
            vec1[iii] = 0;
        for (unsigned j = 0; j < dc[rowid]; j++) {
            vec1[Mc[rowid][j]] = checkVal[rowid][j];
        }
        // check HH^T=0
        for (unsigned i = 0; i < M; i++) {
            for (unsigned iii = 0; iii < N; iii++)
                vec2[iii] = 0;
            for (unsigned j = 0; j < dc[i]; j++) {
                vec2[Mc[i][j]] = checkVal[i][j];
            }
            bool syn_check = false;
            for (unsigned j = 0; j < N; j++) {
                syn_check = trace_inner_product(vec1[j], vec2[j]) ? !syn_check : syn_check;
            }
            if (syn_check) {
                throw std::runtime_error("check_symplectic: syn_check % 2 != 0");
            }
        }
        // check GH^T=0
        for (unsigned i = 0; i < G_rows; i++) {
            bool syn_check = false;
            for (unsigned j = 0; j < N; j++) {
                syn_check = trace_inner_product(vec1[j], G[i][j]) ? !syn_check : syn_check;
            }
            if (syn_check) {
                throw std::runtime_error("check_symplectic / GH^T: syn_check % 2 != 0");
            }
        }
    }
    free(vec1);
    free(vec2);
    std::cout << "check symplectic ok" << std::endl;
    return true;
}

inline bool fileReader::trace_inner_product(unsigned int a, unsigned int b) { return !(a == 0 || b == 0 || a == b); }


