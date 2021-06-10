#include <mccl/config/config.hpp>

#include <mccl/tools/parser.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <utility>

using namespace mccl;

int main(int, char**)
{
    int status = 0;

    Parser parse;
    status |= !parse.load_file("./tests/data/Goppa_1101.txt");

    auto H = parse.get_H();
    auto S = parse.get_S();
    size_t n = parse.get_n();
    size_t k = parse.get_k();
    size_t w = parse.get_w();
  	
  	// check n,k,w
  	status |= (n!=1101);
  	status |= (k!=881);
  	status |= (w!=21);
  	
  	// check  H
    status |= (H.rows() != n-k);
    status |= (H.columns() != n);
    for( size_t r = 0; r < n-k; r++ ) {
    	for( size_t c = 0; c < n-k; c++ ) {
    		status |= (H(r,c)!=(r==c));
    	}
    }
    // check S
    status |= (S.columns() != n-k);

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
