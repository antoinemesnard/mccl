#include <mccl/algorithm/mmt_lm.hpp>

MCCL_BEGIN_NAMESPACE

mmt_lm_config_t mmt_lm_config_default;

vec solve_SD_mmt_lm(const cmat_view& H, const cvec_view& S, unsigned int w)
{
    subISDT_mmt_lm subISDT;
    ISD_mmt_lm<> ISD(subISDT);
    
    return solve_SD(ISD, H, S, w);
}

vec solve_SD_mmt_lm(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap)
{
    subISDT_mmt_lm subISDT;
    ISD_mmt_lm<> ISD(subISDT);
    
    subISDT.load_config(configmap);
    ISD.load_config(configmap);
    
    return solve_SD(ISD, H, S, w);
}

MCCL_END_NAMESPACE