#include <memory>
#include <iostream>
#include <vector>

#ifndef Binner_GPU_h
#define Binner_GPU_h

#include "RecoLocalCalo/HGCalRecProducers/interface/GPUHist2D.h"

struct RecHitGPU { 
        unsigned int index;

        double x;
        double y;
        unsigned int layer;
        double eta;
        double phi;

        double weight;
        double rho;
        double delta;

        int nearestHigher;
        
        int clusterIndex;

        GPU::VecArray<int,20> followers;

        float sigmaNoise;
        float thickness;   

        bool operator > (const RecHitGPU& rhs) const {
                return (rho > rhs.rho);
        }
};

typedef std::vector<RecHitGPU>       LayerRecHitsGPU;
typedef std::vector<LayerRecHitsGPU> HgcRecHitsGPU  ;

namespace BinnerGPU {
    // eta_width = 0.05
    // phi_width = 0.05
    // 2*pi/0.05 = 125
    // 1.4/0.05 = 28
    // 20 (as heuristic)

const int X_BINS=100;
const int Y_BINS=100;
const int MAX_DEPTH=40;

typedef histogram2D<int, X_BINS, Y_BINS, MAX_DEPTH> Histo2D;

Histo2D computeBins(LayerRecHitsGPU layerData);
}


#endif
