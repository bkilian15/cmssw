//#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h

//GPU Add
#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include <math.h>

using namespace BinnerGPU;

namespace HGCalRecAlgosBen{


static const unsigned int lastLayerEE = 28;
static const unsigned int lastLayerFH = 40;

__global__ void kernel_compute_density_ben(Histo2D* theHist, RecHitGPU* theHits, float \
delta_c, int theHitsSize)
{

      size_t idOne = threadIdx.x;
      
      if (idOne < theHitsSize){
            std::array<int, 4> search_box = theHist->searchBox(theHits[idOne].x-delta_c,
                  theHits[idOne].x + delta_c, theHits[idOne].y - delta_c, theHits[idOne].y + delta_c);
            // for(int xBin = search_box[0]; xBin < search_box[1]; xBin++){
            //       for(int yBin = search_box[2]; yBin = search_box[3]){

            //       }
            // }

      }


      // // code from GPU function
      // size_t binIndex = threadIdx.x+blockIdx.x*blockDim.x;
      // size_t binSize = theHist->data_[binIndex].size();
      // for(unsigned int i = 0; i < binSize; i++)
      // {
      //       int idOne = (theHist->data_[binIndex])[i];
      //       for(unsigned int j = 0; j < binSize; j++)
      //       {
      //             int idTwo = (theHist->data_[binIndex])[j];
      //             const double dx = theHits[idOne].x - theHits[idTwo].x;
      //             const double dy = theHits[idOne].y - theHits[idTwo].y;
      //             double distanceGPU = sqrt(dx*dx + dy*dy);
      //             if(distanceGPU < delta_c){
      //                   theHits[idOne].rho += theHits[idTwo].weight;
      //             }
      //       }
                  
      // }
}


double calculateLocalDensity_BinGPU(const BinnerGPU::Histo2D& theHist, LayerRecHitsGPU& theHits, 
      const unsigned int layer, std::vector<double> vecDeltas_){

      double maxdensity = 0.;
      float delta_c;

      if ( layer <= lastLayerEE )
            delta_c = vecDeltas_[0];
      else if ( layer <= lastLayerFH )
            delta_c = vecDeltas_[1];
      else
            delta_c = vecDeltas_[2];
      
      const int numBins = theHist.data_.size();

      // make host items
      RecHitGPU *hInputRecHits;
      hInputRecHits = theHits.data();

      // make device items
      Histo2D *dInputHist; // make GPU histogram
      RecHitGPU *dInputRecHits;  // make input hits for GPU

      cudaMalloc(&dInputHist, sizeof(Histo2D));
      cudaMalloc(&dInputRecHits, sizeof(RecHitGPU)*theHits.size());

      cudaMemcpy(dInputHist, &theHist, sizeof(Histo2D), cudaMemcpyHostToDevice);
      cudaMemcpy(dInputRecHits, hInputRecHits, sizeof(RecHitGPU)*theHits.size(), cudaMemcpyHostToDe\
vice);
      std::cout << " mem copied ";
     

      //RecHitGPU dInputRecHitsArray[theHits.size()];
      std::cout << "about to call kernel";
      // KERNEL CALL
      const dim3 blockSize(50, 1, 1);
      const dim3 gridSize(50, 1, 1);
      kernel_compute_density_ben <<<gridSize, blockSize>>>(dInputHist, dInputRecHits, delta_c, theHist.data_.size());
      std::cout << "  finished kernel call" << std::endl;
      cudaMemcpy(hInputRecHits, dInputRecHits, sizeof(RecHitGPU)*theHits.size(), cudaMemcpyDeviceToHost);

      cudaFree(dInputHist);
      cudaFree(dInputRecHits);

      std::cout << "Inside GPU " << std::endl;
      for(unsigned int j = 0; j< theHits.size(); j++) {
            std::cout << hInputRecHits[j].rho << " " ;
            if (maxdensity < hInputRecHits[j].rho) {
                  maxdensity = hInputRecHits[j].rho;
      }
    }
      return maxdensity;

}

}
