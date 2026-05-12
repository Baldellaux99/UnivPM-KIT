#include <iostream>
#include "stabilizerCodes.h"
int main() {
    int n = 21;
    int k = 3;
    int m = 32;
    fileReader matrix_supplier(n, k, m);
    stabilizerCodes code(n, k, m, matrix_supplier);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
