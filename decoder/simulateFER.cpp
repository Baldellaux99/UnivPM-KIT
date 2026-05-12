#include "stabilizerCodes.h"

#include "helpers.h"

#include <iostream>
#include <omp.h>

int main(int argc, char **argv)
{
    //    int n = 50;
    //    int k = 12;
    //    int m = 42; //[244,40]]
    //code parameters
    int n = 128;
    int k = 36;
    int m = 128;

    // decodingOpt: 0 for flooding BP, 1 for genie-aided BP, 2 for CAMEL
    int decodingOpt = 2;

    int number_guess = 1;
    constexpr int max_frame_errors = 300;
    constexpr int max_decoded_words = 100000000;
    double defualt_ep0 = 0.1;

    double start = 0.1;
    double end = 0.01;
    // double step = -0.001;

    // auto ep_list = helpers::adaptive_decade_steps(start, end);
    // double ep_list[num_sim_points];

    // for (int i = 0; i < num_sim_points; ++i)
    // {
    //     ep_list[i] = start + i * step;
    // }

    // double ep_list[] = {0.12, 0.11, 0.1, 0.09};
    double ep_list[] = {0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.018,0.017,0.016 ,0.14,0.13,0.12,0.11,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01};//, 0.016, 0.014, 0.012};
    //        std::vector<double> ep_list(0);
    //        for (double ep = 0.1; ep>0.001; ep/=1.5){
    //            ep_list.push_back(ep);
    //        }
    //    0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,
    //    const std::vector<double> ep_list{0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02, 0.01
    //            0.019,0.018,0.017,0.016 0.14,0.13,0.12,0.11,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01
    //    };
    //    const std::vector<double> ep_list{
    //            0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02, 0.01
    //    };

    //    std::vector<double> ep_list;
    //    for (int i=0; i<9; i++){
    //        ep_list.push_back(0.05 / double (1<<i));
    //    }
    //
    int decIterNum = 15;
    double ep0 = 0.1;

    fileReader matrix_supplier(n, k, m);
    matrix_supplier.check_symplectic();

    std::cout << "% collect " << max_frame_errors << " frame errors or " << max_decoded_words
              << " decoded error patterns\n";
    std::cout << "% depolarizing prob. list to be simulated: ";
    for (auto ep : ep_list)
    {
        std::cout << ep << ", ";
    }
    std::cout << std::endl;
    if (decodingOpt == 0)
    {
        std::cout << "% Decoding option: Flooding BP\n";
    }
    else if (decodingOpt == 1)
    {
        std::cout << "% Decoding option: Genie-aided BP\n";
    }
    else if (decodingOpt == 2)
    {
        std::cout << "% Decoding option: CAMEL\n";
    }

    std::cout << "% [[" << n << "," << k << +"]], " << m << " checks, " << decIterNum << " iter, ep0=" << ep0 << std::endl;

    for (double epsilon : ep_list)
    {
        //            epsilon*=1.5;
        double total_decoding = 0;
        double failure = 0;

#pragma omp parallel
        {
            while (failure <= max_frame_errors && total_decoding <= max_decoded_words)
            {
                //                    stabilizerCodes code(n, k, m, matrix_supplier);
                stabilizerCodes code(n, k, m, matrix_supplier);

                // 0 for flooding, 1 for genie-aided, 2 for multi-path decoding
                code.DecodingOption = decodingOpt;

                code.add_error_given_epsilon(epsilon);
                //                    code.error.clear();
                //                    code.errorString.clear();
                //                    code.error_weight=0;
                //
                //                    for (unsigned i = 0; i < code.N; i++) {
                //                        if (i==code.N-1){
                //                            code.error.push_back(3);
                //                            code.errorString.emplace_back("Y" + std::to_string(i)); code.error_weight+=1;
                //                        }else{
                //                            code. error.push_back(0);
                //                        }
                //
                //                    }
                //                    stabilizerCodes code2(code);
                //                    code2.print_msg = false;
                //                    code2.DecodingOption = 1;
                //                    auto success2 = code2.decode(decIterNum, ep0, 1);
                auto success = code.decode(decIterNum, ep0, number_guess);

                //                    auto success = code.binary_decode(decIterNum, ep0, true);

                //                    if (!success && !code.error_hat.empty()){
                //                        std::cout<<"true error"<<std::endl;
                //                        for (int i=0; i<code.error_weight; i++){
                //                            std::cout<<" "<<code.errorString[i];
                //                        }

                //                        std::cout<<std::endl<<"genie-aided estimation"<<std::endl;
                //                        for (int i=0; i<n; i++){
                //                            if (code2.error_hat[i]==1) std::cout<<" X"<<i;
                //                            if (code2.error_hat[i]==2) std::cout<<" Z"<<i;
                //                            if (code2.error_hat[i]==3) std::cout<<" Y"<<i;
                //                        }

                //                        std::cout<<std::endl<<" estimation"<<std::endl;
                //                        for (int i=0; i<n; i++){
                //                            if (code.error_hat[i]==1) std::cout<<" X"<<i;
                //                            if (code.error_hat[i]==2) std::cout<<" Z"<<i;
                //                            if (code.error_hat[i]==3) std::cout<<" Y"<<i;
                //                        }
                //
                //                        std::cout<<"\n==================="<<std::endl;
                //                    }

//                success = code.LUTdecode();
//                for (int i=0; i<n; i++){
//                    std::cout<<code.error_hat[i];
//                }
//                std::cout<<std::endl;
#pragma omp critical
                {
                    if (!success)
                        failure += 1;
                    total_decoding += 1;
                    if ((int)total_decoding % 10000000 == 0)
                    {
                        std::cout << "% FE " << failure << ", total dec. " << total_decoding << "\\\\" << std::endl;
                    }
                }
            }
        }
        std::cout << "% FE " << failure << ", total dec. " << total_decoding << "\\\\" << std::endl;
        std::cout << epsilon << " " << (failure / total_decoding) << "\\\\" << std::endl;
    }

    return 0;
}
