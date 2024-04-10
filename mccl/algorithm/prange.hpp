#ifndef MCCL_ALGORITHM_PRANGE_HPP
#define MCCL_ALGORITHM_PRANGE_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>

MCCL_BEGIN_NAMESPACE

class subISDT_prange
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    subISDT_prange() : stats("Prange")
    {

    } 

    // API member function
    ~subISDT_prange() final
    {
    }
    
    void load_config(const configmap_t&) final
    {
    }
    void save_config(configmap_t&) final
    {
    }
    
    // API member function
    void initialize(const cmat_view&, size_t H2Tcolumns, const cvec_view&, unsigned int, callback_t _callback, void* _ptr) final
    {
        stats.cnt_initialize.inc();
        stats.time_initialize.start();
        // should only be used with l=0
        if (H2Tcolumns != 0)
            throw std::runtime_error("subISDT_prange::initialize(): Prange doesn't support l>0");
        callback = _callback;
        ptr = _ptr;
        stats.time_initialize.stop();
    }
    
    // API member function
    void prepare_loop() final
    {
        stats.cnt_prepare_loop.inc();
    }
    
    // API member function
    bool loop_next() final
    {
        stats.cnt_loop_next.inc();
        stats.time_loop_next.start();        
        (*callback)(ptr, nullptr, nullptr, 0);
        stats.time_loop_next.stop();
        return false;
    }
    
    // API member function
    void solve() final
    {
        stats.cnt_solve.inc();
        stats.time_solve.start();
        loop_next();
        stats.time_solve.stop();
    }
    decoding_statistics get_stats() const { return stats; };
private:
    callback_t callback;
    void* ptr;
    decoding_statistics stats;
};

template<size_t _bit_alignment = 64>
using ISD_prange = ISD_generic<subISDT_prange,_bit_alignment>;

vec solve_SD_prange(const cmat_view& H, const cvec_view& S, unsigned int w);
static inline vec solve_SD_prange(const syndrome_decoding_problem& SD)
{
    return solve_SD_prange(SD.H, SD.S, SD.w);
}

vec solve_SD_prange(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap);
static inline vec solve_SD_prange(const syndrome_decoding_problem& SD, const configmap_t& configmap)
{
    return solve_SD_prange(SD.H, SD.S, SD.w, configmap);
}

MCCL_END_NAMESPACE

#endif
