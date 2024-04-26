#ifndef MCCL_ALGORITHM_MMT_LM_HPP
#define MCCL_ALGORITHM_MMT_LM_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/tools/unordered_multimap.hpp>
#include <mccl/tools/bitfield.hpp>
#include <mccl/tools/enumerate.hpp>
#include <mccl/tools/utils.hpp>

MCCL_BEGIN_NAMESPACE

struct mmt_lm_config_t
{
    const std::string modulename = "mmt_lm";
    const std::string description = "MMT low memory configuration";
    const std::string manualstring =
        "MMT_LM:\n"
        "\tParameters: p, l2, A"
        "\tAlgorithm:\n"
        "\t\tTODO\n"
        ;

    unsigned int p = 8;
    unsigned int l2 = 10;
    unsigned int A = 4;

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 8, "subISDT parameter p");
        c(l2, "l2", 10, "subISDT parameter l2");
        c(A, "A", 4, "subISDT parameter A");
    }
};

// global default. modifiable.
// at construction of subISDT_mmt_lm the current global default values will be loaded
extern mmt_lm_config_t mmt_lm_config_default;



class subISDT_mmt_lm
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    // API member function
    ~subISDT_mmt_lm() final
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

    subISDT_mmt_lm()
        : config(mmt_lm_config_default), stats("MMT-LM")
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
        l2 = config.l2;
        A = config.A;
        // set attack parameters
        p1 = p/2; p2 = p - p1;
        p11 = p/4; p12 = p1 - p11;
        l1 = columns - l2;
        rows = H12T.rows();
        rows1 = rows/2; rows2 = rows - rows1;

        // check configuation
        // TODO

        firstwordmask = detail::lastwordmask(columns);
        padmask = ~firstwordmask;
        syndmask = detail::lastwordmask(l2);

        hashmap12.define_keymask(syndmask);
        hashmap.define_keymask(firstwordmask);

        // compute reasonable reserve sizes
        double L1 = detail::binomial<double>(rows2, p12);
        double L2 = L1 * L1 / pow(2.0, double(l2));
        // double L3 = L2 * L2 / pow(2.0, double(l1));
        hashmap12.clear();
        hashmap.clear();
        hashmap12.reserve(size_t(L1));
        hashmap.reserve(size_t (L2));

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

    //API member function
    void prepare_loop() final
    {        
        stats.cnt_prepare_loop.inc();
        stats.time_prepare_loop.start();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_prepareloop);

        firstwords.resize(rows);
        for (unsigned i = 0; i < rows; ++i)
            firstwords[i] = (*H12T.word_ptr(i));
        Sval = (*S.word_ptr());
        
        hashmap12.clear();

        stats.time_other_1.start();
        enumerate.enumerate(firstwords.data()+0, firstwords.data()+rows2, p12,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                uint64_t packed_indices = pack_indices(idxbegin, idxend);
                hashmap12.insert(val, packed_indices);
            });
        hashmap12.finalize_insert();
        stats.time_other_1.stop();

        a = 0;

        state = true;

        stats.time_prepare_loop.stop();
    }

    // API member function
    bool loop_next() final
    {
        stats.cnt_loop_next.inc();
        stats.time_loop_next.start();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);

        hashmap.clear();
        
        stats.time_other_2.start();
        enumerate.enumerate(firstwords.data()+rows2, firstwords.data()+rows, p11,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val1)
            {
                val1 ^= Sval;
                uint32_t* it = idx+0;
                for (auto it2 = idxbegin; it2 != idxend; ++it2, ++it)
                    *it = *it2 + rows2;
                uint64_t packed_indices1 = pack_indices(idx+0, it);
                hashmap12.queue_match(val1 ^ a, packed_indices1,
                    [this](const uint64_t val1, const uint64_t packed_indices1, const uint64_t val2, const uint64_t packed_indices2)
                    {
                        uint64_t val = val1 ^ val2;
                        pair_uint64_t packed_indices = { packed_indices1, packed_indices2 };
                        hashmap.insert(val, packed_indices);
                    });
            });
        hashmap12.finalize_match(
            [this](const uint64_t val1, const uint64_t packed_indices1, const uint64_t val2, const uint64_t packed_indices2)
            {
                uint64_t val = val1 ^ val2;
                pair_uint64_t packed_indices = { packed_indices1, packed_indices2 };
                hashmap.insert(val, packed_indices);
            });
        hashmap.finalize_insert();
        stats.time_other_2.stop();
        
        stats.time_other_3.start();
        enumerate.enumerate(firstwords.data()+rows2, firstwords.data()+rows, p11,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val11)
            {
                uint32_t* it = idx+0;
                for (auto it2 = idxbegin; it2 != idxend; ++it2, ++it)
                    *it = *it2 + rows2;
                uint64_t packed_indices11 = pack_indices(idx+0, it);
                hashmap12.queue_match(val11 ^ a, packed_indices11,
                    [this](const uint64_t val11, const uint64_t packed_indices11, const uint64_t val12, const uint64_t packed_indices12)
                    {
                        uint64_t val1 = val11 ^ val12;
                        pair_uint64_t packed_indices1 = { packed_indices11, packed_indices12 };
                        hashmap.queue_match(val1, packed_indices1, process_candidate);
                        return state;
                    });
            return state;
            });
        hashmap12.finalize_match(
            [this](const uint64_t val11, const uint64_t packed_indices11, const uint64_t val12, const uint64_t packed_indices12)
            {
                uint64_t val1 = val11 ^ val12;
                pair_uint64_t packed_indices1 = { packed_indices11, packed_indices12 };
                hashmap.queue_match(val1, packed_indices1, process_candidate);
                return state;
            });
        hashmap.finalize_match(process_candidate);
        stats.time_other_3.stop();

        ++a;
        state = state && (a < A);
        stats.time_loop_next.stop();
        return state;
    }

    std::function<bool(const uint64_t, const pair_uint64_t, const uint64_t, const pair_uint64_t)> process_candidate =
        [this](const uint64_t val1, const pair_uint64_t packed_indices1, const uint64_t val2, const pair_uint64_t packed_indices2)
        {
            auto it = unpack_indices2(packed_indices1.first, packed_indices2.first, idx+0);
            auto it2 = unpack_indices2(packed_indices1.second, packed_indices2.second, it);

            size_t p0 = it2 - idx;
            if (p0 == p)
                stats.cnt_L0_0.inc();
            else if (p0 == p - 2)
                stats.cnt_L0_1.inc();
            else
                stats.cnt_L0_2.inc();

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

    uint32_t* unpack_indices2(uint64_t x1, uint64_t x2, uint32_t* first)
    {
        size_t i = 0, j = 0;
        while ((i < 4) && (j < 4))
        {   uint32_t y1 = uint32_t(x1 & 0xFFFF);
            uint32_t y2 = uint32_t(x2 & 0xFFFF);
            if ((y1 == 0xFFFF) || (y2 == 0xFFFF))
                break;
            if (y1 == y2)
            {   x1 >>= 16;
                x2 >>= 16;
                ++i;
                ++j;
                continue; }
            if ((y1 > y2))
            {   *first = y1;
                x1 >>= 16;
                ++i; }
            else
            {   *first = y2;
                x2 >>= 16;
                ++j; }
            ++first; }
        for (; i < 4; ++i)
        {   uint32_t y1 = uint32_t(x1 & 0xFFFF);
            if (y1 == 0xFFFF)
                break;
            *first = y1;
            ++first;
            x1 >>= 16; }
        for (; j < 4; ++j)
        {   uint32_t y2 = uint32_t(x2 & 0xFFFF);
            if (y2 == 0xFFFF)
                break;
            *first = y2;
            ++first;
            x2 >>= 16; }
        return first;
    }

    decoding_statistics get_stats() const { return stats; };

    void reset_stats() { stats.reset(); };

    double get_inverse_proba()
    {
        size_t k = rows - columns;
        size_t n = H12T.columns() + k;
        double sum = 0;
        sum += detail::binomial<double>(n - k - columns, wmax - p) * stats.cnt_L0_0.mean();
        sum += detail::binomial<double>(n - k - columns, wmax - p + 2) * stats.cnt_L0_1.mean();
        sum += detail::binomial<double>(n - k - columns, wmax - p + 4) * stats.cnt_L0_2.mean();
        return std::min<double>(std::pow(2.0, double(n - k)), detail::binomial<double>(n, wmax)) / (std::pow(2.0, double(columns)) * sum);
    }

    void optimize_parameters(size_t k, unsigned int& config_l, std::function<bool()> run_test)
    {
        unsigned int lopt = config_l, ltest;
        unsigned int popt = config.p, ptest;
        unsigned int l2opt = config.l2, l2test;
        unsigned int Aopt = config.A, Atest;
        double power;
        for (ptest = 4; ptest <= 16; ptest += 4)
        {
            if (k & 1) { ltest = 1; power = 2.0 * std::sqrt(2.0); }
            else { ltest = 2; power = 4.0; }
            while (power <= detail::binomial<double>((k + ltest + 2) / 2, ptest / 4))
            {
                ltest += 2;
                power *= 2.0;
            }
            l2test = 1;
            power = 4.0;
            while (power <= detail::binomial<double>((k + ltest) / 2, ptest / 4))
            {
                l2test += 1;
                power *= 2.0;
            }
            Atest = 1;
            config_l = ltest;
            config.p = ptest;
            config.l2 = l2test;
            config.A = Atest;
            if (run_test())
            {
                lopt = ltest;
                popt = ptest;
                l2opt = l2test;
                Aopt = Atest;
            }
        }
        config_l = lopt;
        config.p = popt;
        config.l2 = l2opt;
        config.A = Aopt;
    }

private:
    callback_t callback;
    void* ptr;
    cmat_view H12T;
    cvec_view S;
    size_t columns, words;
    unsigned int wmax;

    batch_unordered_multimap<uint64_t, uint64_t, uint64_t, true> hashmap12;
    batch_unordered_multimap<uint64_t, pair_uint64_t, pair_uint64_t, true> hashmap;

    enumerate_t<uint32_t> enumerate;
    uint32_t idx[16];

    std::vector<uint64_t> firstwords;
    uint64_t firstwordmask, padmask, syndmask, Sval;

    uint64_t a;
    bool state;

    size_t p, p1, p2, p11, p12, rows, rows1, rows2, l1, l2, A;

    mmt_lm_config_t config;
    decoding_statistics stats;
    cpucycle_statistic cpu_prepareloop, cpu_loopnext, cpu_callback;
};



template<size_t _bit_alignment = 64>
using ISD_mmt_lm = ISD_generic<subISDT_mmt_lm,_bit_alignment>;

vec solve_SD_mmt_lm(const cmat_view& H, const cvec_view& S, unsigned int w);
static inline vec solve_SD_mmt_lm(const syndrome_decoding_problem& SD)
{
    return solve_SD_mmt_lm(SD.H, SD.S, SD.w);
}

vec solve_SD_mmt_lm(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap);
static inline vec solve_SD_mmt_lm(const syndrome_decoding_problem& SD, const configmap_t& configmap)
{
    return solve_SD_mmt_lm(SD.H, SD.S, SD.w, configmap);
}



MCCL_END_NAMESPACE

#endif