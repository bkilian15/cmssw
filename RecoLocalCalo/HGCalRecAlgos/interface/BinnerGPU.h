#include <memory>
#include <iostream>
#include <vector>

#ifndef Binner_GPU_h
#define Binner_GPU_h

#include "RecoLocalCalo/HGCalRecAlgos/interface/GPUHist2D.h"

struct RecHitGPU { // rename RechitForBinning --> RecHitGPU
        unsigned int index;
        float eta;
        float phi;
};

typedef std::vector<RecHitGPU>       LayerRecHitsGPU; // rename LayerData --> LayerRecHitsGPU
typedef std::vector<LayerRecHitsGPU> HgcRecHitsGPU  ; // rename BinningData --> HgcRecHitsGPU

namespace BinnerGPU {
    // eta_width = 0.05
    // phi_width = 0.05
    // 2*pi/0.05 = 125
    // 1.4/0.05 = 28
    // 20 (as heuristic)

const int ETA_BINS=28;
const int PHI_BINS=126;
const int MAX_DEPTH=20;



//std::shared_ptr<int> 
histogram2D<int, ETA_BINS, PHI_BINS, MAX_DEPTH> computeBins(LayerRecHitsGPU layerData);

}


#endif
