#include <memory>
#include <iostream>
#include <vector>

#ifndef Binner_GPU_h
#define Binner_GPU_h

#include "RecoLocalCalo/HGCalRecAlgos/interface/GPUHist2D.h"

struct RecHitGPU { 
        unsigned int index;
        float eta;
        float phi;
};

typedef std::vector<RecHitGPU>       LayerRecHitsGPU;
typedef std::vector<LayerRecHitsGPU> HgcRecHitsGPU  ;

namespace BinnerGPU {
    // eta_width = 0.05
    // phi_width = 0.05
    // 2*pi/0.05 = 125
    // 1.4/0.05 = 28
    // 20 (as heuristic)

const int ETA_BINS=28;
const int PHI_BINS=126;
const int MAX_DEPTH=20;

typedef histogram2D<int, ETA_BINS, PHI_BINS, MAX_DEPTH> Histo2D;

Histo2D computeBins(LayerRecHitsGPU layerData);

}


#endif
