#ifndef MCCL_ALGORITHM_STERN_DUMER_HPP
#define MCCL_ALGORITHM_STERN_DUMER_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/tools/unordered_multimap.hpp>
#include <mccl/tools/bitfield.hpp>
#include <mccl/tools/enumerate.hpp>
#include <mccl/tools/utils.hpp>

MCCL_BEGIN_NAMESPACE

struct stern_dumer_config_t
{
    const std::string modulename = "stern_dumer";
    const std::string description = "Stern/Dumer configuration";
    const std::string manualstring = 
        "Stern/Dumer:\n"
        "\tParameters: p\n"
        "\tAlgorithm:\n"
        "\t\tPartition columns of H2 into two sets.\n\t\tCompare p/2-columns sums from both sides.\n\t\tReturn pairs that sum up to S2.\n"
        ;

    unsigned int p = 4;

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 4, "subISDT parameter p");
    }
};

// global default. modifiable.
// at construction of subISDT_stern_dumer the current global default values will be loaded
extern stern_dumer_config_t stern_dumer_config_default;



class subISDT_stern_dumer
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    // API member function
    ~subISDT_stern_dumer() final
    {
        cpu_prepareloop.refresh();
        cpu_loopnext.refresh();
        cpu_callback.refresh();
        if (cpu_loopnext.total() > 0)
        {
            std::cerr << "prepare : " << cpu_prepareloop.total() << std::endl;
            std::cerr << "nextloop: " << cpu_loopnext.total() - cpu_callback.total() << std::endl;
            std::cerr << "callback: " << cpu_callback.total() << std::endl;
        }
    }
    
    subISDT_stern_dumer()
        : config(stern_dumer_config_default), stats("Stern/Dumer")
    {
    }

    void load_config(const configmap_t& configmap) final
    {
        mccl::load_config(config, configmap);
    }
    void save_config(configmap_t& configmap) final
    {
        mccl::save_config(config, configmap);
    }

    // API member function
    void initialize(const cmat_view& _H12T, size_t _H2Tcolumns, const cvec_view& _S, unsigned int w, callback_t _callback, void* _ptr) final
    {
        stats.cnt_initialize.inc();
        stats.time_initialize.start();

        // copy initialization parameters
        H12T.reset(_H12T);
        S.reset(_S);
        columns = _H2Tcolumns;
        callback = _callback;
        ptr = _ptr;
        wmax = w;

        // copy parameters from current config
        p = config.p;
        // set attack parameters
        p1 = p/2; p2 = p - p1;
        rows = H12T.rows();
        rows1 = rows/2; rows2 = rows - rows1;

        words = (columns+63)/64;

        // check configuration
        if (p < 2)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support p < 2");
        if (columns < 6)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support l < 6 (since we use bitfield)");
        if (words > 1)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support l > 64 (yet)");
        if ( p > 8)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support p > 8 (yet)");
        if (rows1 >= 65535 || rows2 >= 65535)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support rows1 or rows2 >= 65535");

        firstwordmask = detail::lastwordmask(columns);
        padmask = ~firstwordmask;
        
        bitfield.resize(columns);

        hashmap.define_keymask(firstwordmask);

        // compute a reasonable reserve size
        double L1 = detail::binomial<double>(rows2, p2);
        double L2 = L1 * L1 / pow(2.0, double(columns));
        hashmap.clear();
        hashmap.reserve(size_t(std::min<double>(L1, L2)), 1.0f);

        stats.time_initialize.stop();
    }

    // API member function
    void solve() final
    {
        stats.cnt_solve.inc();
        stats.time_solve.start();
        prepare_loop();
        while (loop_next())
            ;
        stats.time_solve.stop();
        stats.refresh();
    }
    
    // API member function
    void prepare_loop() final
    {
        stats.cnt_prepare_loop.inc();
        stats.time_prepare_loop.start();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_prepareloop);
        
        firstwords.resize(rows);
        for (unsigned i = 0; i < rows; ++i)
            firstwords[i] = (*H12T.word_ptr(i));
        Sval = (*S.word_ptr());
        
        bitfield.clear();
        hashmap.clear();

        state = true;

        stats.time_prepare_loop.stop();
    }

    // API member function
    bool loop_next() final
    {
        stats.cnt_loop_next.inc();
        stats.time_loop_next.start();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);

        // stage 1: store left-table in bitfield
        stats.time_other_1.start();
        enumerate.enumerate_val(firstwords.data()+rows2, firstwords.data()+rows, p1,
            [this](uint64_t val)
            { 
                stats.cnt_L11.inc();
                bitfield.stage1(val); 
            });
        stats.time_other_1.stop();
        // stage 2: compare right-table with bitfield: store matches
        // note we keep the packed indices at offset 0 in firstwords for right-table
        stats.time_other_2.start();
        enumerate.enumerate(firstwords.data()+0, firstwords.data()+rows2, p2,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                stats.cnt_L12.inc();
                val ^= Sval;
                if (bitfield.stage2(val))
                    hashmap.insert(val, pack_indices(idxbegin,idxend) );
            });
        hashmap.finalize_insert();
        stats.time_other_2.stop();
        // stage 3: retrieve matches from left-table and process
        stats.time_other_3.start();
        enumerate.enumerate(firstwords.data()+rows2, firstwords.data()+rows, p1,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val1)
            {
                if (bitfield.stage3(val1))
                {
                    uint32_t* it = idx+0;
                    // note that left-table indices are offset rows2 in firstwords
                    for (auto it2 = idxbegin; it2 != idxend; ++it2,++it)
                        *it = *it2 + rows2;
                    uint64_t packed_indices1 = pack_indices(idx+0, it);
                    hashmap.queue_match(val1, packed_indices1, process_candidate);
                }
                return state;
            });
        hashmap.finalize_match(process_candidate);
        stats.time_other_3.stop();
        stats.time_loop_next.stop();
        return false;
    }

    std::function<bool(const uint64_t, const uint64_t, const uint64_t, const uint64_t)> process_candidate =
        [this](const uint64_t val1, const uint64_t packed_indices1, const uint64_t val2, const uint64_t packed_indices2)
        {
            stats.cnt_L0_0.inc();
            
            auto it = unpack_indices(packed_indices1, idx+0);
            auto it2 = unpack_indices(packed_indices2, it);
            unsigned int w = hammingweight((val1 ^ val2) & padmask);

            MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
            if (!(*callback)(ptr, idx+0, it2, w))
                state = false;
            return state;
        };
    
    static uint64_t pack_indices(const uint32_t* begin, const uint32_t* end)
    {
        uint64_t x = ~uint64_t(0);
        for (; begin != end; ++begin)
        {
            x <<= 16;
            x |= uint64_t(*begin);
        }
        return x;
    }
    
    uint32_t* unpack_indices(uint64_t x, uint32_t* first)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            uint32_t y = uint32_t(x & 0xFFFF);
            if (y == 0xFFFF)
                break;
            *first = y;
            ++first;
            x >>= 16;
        }
        return first;
    }

    decoding_statistics get_stats() const { return stats; };

    void reset_stats() { stats.reset(); };

    double get_inverse_proba()
    {
        size_t k = rows - columns;
        size_t n = H12T.columns() + k;
        return std::min<double>(std::pow(2.0, double(n - k)), detail::binomial<double>(n, wmax)) / (detail::binomial<double>(n - k - columns, wmax - p) * std::pow(2.0, double(columns)) * double(stats.cnt_L0_0.mean()));
    }

    void optimize_parameters(size_t k, unsigned int& config_l, std::function<bool()> run_test)
    {
        unsigned int lopt = config_l, ltest;
        unsigned int popt = config.p, ptest;
        double power;
        for (ptest = 2; ptest <= 8; ptest += 2)
        {
            if (k & 1) { ltest = 7; power = 512.0; }
            else { ltest = 6; power = 256.0; }
            while (power <= detail::binomial<double>((k + ltest + 2) / 2, ptest / 2))
            {
                ltest += 2;
                power *= 4.0;
            }
            config_l = ltest;
            config.p = ptest;
            if (run_test())
            {
                lopt = ltest;
                popt = ptest;
            }
        }
        config_l = lopt;
        config.p = popt;
    }

private:
    callback_t callback;
    void* ptr;
    cmat_view H12T;
    cvec_view S;
    size_t columns, words;
    unsigned int wmax;
    
    staged_bitfield<false,false> bitfield;
    batch_unordered_multimap<uint64_t, uint64_t, uint64_t, true> hashmap;
    
    enumerate_t<uint32_t> enumerate;
    uint32_t idx[16];

    std::vector<uint64_t> firstwords;
    uint64_t firstwordmask, padmask, Sval;

    bool state;
    
    size_t p, p1, p2, rows, rows1, rows2;
    
    stern_dumer_config_t config;
    decoding_statistics stats;
    cpucycle_statistic cpu_prepareloop, cpu_loopnext, cpu_callback;
};



template<size_t _bit_alignment = 64>
using ISD_stern_dumer = ISD_generic<subISDT_stern_dumer,_bit_alignment>;

vec solve_SD_stern_dumer(const cmat_view& H, const cvec_view& S, unsigned int w);
static inline vec solve_SD_stern_dumer(const syndrome_decoding_problem& SD)
{
    return solve_SD_stern_dumer(SD.H, SD.S, SD.w);
}

vec solve_SD_stern_dumer(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap);
static inline vec solve_SD_stern_dumer(const syndrome_decoding_problem& SD, const configmap_t& configmap)
{
    return solve_SD_stern_dumer(SD.H, SD.S, SD.w, configmap);
}



MCCL_END_NAMESPACE

#endif
