//#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"

//GPU Add
#include "RecoLocalCalo/HGCalRecProducers/interface/BinnerGPU.h"
#include <math.h>
#include <chrono>

using namespace BinnerGPU;

namespace HGCalRecAlgos{


  static const unsigned int lastLayerEE = 28;
  static const unsigned int lastLayerFH = 40;
  static const float maxDelta = 1000.0; 
  static const unsigned int maxNSeeds = 1024; 
  static const unsigned int BufferSizePerSeed = 40; 




  __device__ double distance2GPU(const RecHitGPU pt1, const RecHitGPU pt2) {
    //distance squared
    const double dx = pt1.x - pt2.x;
    const double dy = pt1.y - pt2.y;
    return (dx*dx + dy*dy);
  } 

  __global__ void kernel_compute_histogram(Histo2D *dOutputData, RecHitGPU *dInputData,  const size_t numRechits) {
    
    size_t rechitLocation = blockIdx.x * blockDim.x + threadIdx.x;

    if(rechitLocation >= numRechits) {
        printf("inside if. recHitsLocation: %d\n", rechitLocation);
        return;
    }

    int layer = dInputData[rechitLocation].layer;
    printf("layer (in 1st kernel): %d\n", layer);
    
    float x = dInputData[rechitLocation].x;
    float y = dInputData[rechitLocation].y;
   
    dOutputData[layer].fillBinGPU(x, y, rechitLocation);// dOutputData[layer]->fillBinGPU(x, y, rechitLocation);
    __syncthreads();
  }


  __global__ void kernel_compute_density( Histo2D* theHist, RecHitGPU* theHits, 
                                          float* vecDeltas_,//float delta_c, 
                                          int theHitsSize
                                          ) {
    printf("hello from the 2nd kernel!!!!");
    size_t idOne = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = theHits[idOne].layer;
    printf("layer (in 2nd kernel): %d\n", layer);

    float delta_c;

    if (layer <= lastLayerEE)
      delta_c = vecDeltas_[0];
    else if (layer <= lastLayerFH)
      delta_c = vecDeltas_[1];
    else
      delta_c = vecDeltas_[2];

    if (idOne < theHitsSize){


      int xBinMin = theHist[layer].computeXBinIndex(std::max(float(theHits[idOne].x - delta_c), theHist[layer].limits_[0]));
      int xBinMax = theHist[layer].computeXBinIndex(std::min(float(theHits[idOne].x + delta_c), theHist[layer].limits_[1]));
      int yBinMin = theHist[layer].computeYBinIndex(std::max(float(theHits[idOne].y - delta_c), theHist[layer].limits_[2]));
      int yBinMax = theHist[layer].computeYBinIndex(std::min(float(theHits[idOne].y + delta_c), theHist[layer].limits_[3]));

      // printf("%f %f %f %f \n", xBinMin, xBinMax, yBinMin, yBinMax);

      for(int xBin = xBinMin; xBin < xBinMax+1; ++xBin) {
        for(int yBin = yBinMin; yBin < yBinMax+1; ++yBin) {
          
          size_t binIndex = theHist[layer].getBinIdx_byBins(xBin,yBin);
          size_t binSize  = theHist[layer].data_[binIndex].size();


          for (unsigned int j = 0; j < binSize; j++) {
            int idTwo = (theHist[layer].data_[binIndex])[j];

            double distance = sqrt(distance2GPU(theHits[idOne], theHits[idTwo]));

            if(distance < delta_c) {
              theHits[idOne].rho += (idOne == idTwo ? 1. : 0.5) * theHits[idTwo].weight;
            }
          }
        }
      }
    }
  } //kernel


  __global__ void kernel_compute_distanceToHigher(Histo2D* theHist, RecHitGPU* theHits,
                                                  float* vecDeltas_,//float delta_c, 
                                                  int theHitsSize, 
                                                  float outlierDeltaFactor_
                                                  ) {
    size_t idOne = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = theHits[idOne].layer;
    printf("layer (in 3rd kernel): %d\n", layer);
    
    float delta_c;

    if (layer <= lastLayerEE)
      delta_c = vecDeltas_[0];
    else if (layer <= lastLayerFH)
      delta_c = vecDeltas_[1];
    else
      delta_c = vecDeltas_[2];
    // int temp = theHist->getBinIdx_byBins(1,1);

    if (idOne < theHitsSize){
      // initialize delta and nearest higer for i
      float i_delta = maxDelta;
      int i_nearestHigher = -1;


      // get search box for ith hit
      // garrantee to cover "outlierDeltaFactor_*delta_c"
      int xBinMin = theHist[layer].computeXBinIndex(std::max(float(theHits[idOne].x - outlierDeltaFactor_*delta_c), theHist[layer].limits_[0]));
      int xBinMax = theHist[layer].computeXBinIndex(std::min(float(theHits[idOne].x + outlierDeltaFactor_*delta_c), theHist[layer].limits_[1]));
      int yBinMin = theHist[layer].computeYBinIndex(std::max(float(theHits[idOne].y - outlierDeltaFactor_*delta_c), theHist[layer].limits_[2]));
      int yBinMax = theHist[layer].computeYBinIndex(std::min(float(theHits[idOne].y + outlierDeltaFactor_*delta_c), theHist[layer].limits_[3]));
      // loop over all bins in the search box
      for(int xBin = xBinMin; xBin < xBinMax+1; ++xBin) {
        for(int yBin = yBinMin; yBin < yBinMax+1; ++yBin) {
          
          // get the id of this bin
          size_t binIndex = theHist[layer].getBinIdx_byBins(xBin,yBin);
          // get the size of this bin
          size_t binSize  = theHist[layer].data_[binIndex].size();

          // loop over all hits in this bin
          for (unsigned int j = 0; j < binSize; j++) {
            int idTwo = (theHist[layer].data_[binIndex])[j];

            float distance = sqrt(distance2GPU(theHits[idOne], theHits[idTwo]));
            bool foundHigher = theHits[idTwo].rho > theHits[idOne].rho;
            
            if(foundHigher && distance <= i_delta) {
              // update i_delta
              i_delta = distance;
              // update i_nearestHigher
              i_nearestHigher = idTwo;
            }
          }
        }
      }
      
      bool foundNearestHigherInSearchBox = (i_delta != maxDelta);
      // if i is not a seed or noise
      if (foundNearestHigherInSearchBox){
        // pass i_delta and i_nearestHigher to ith hit
        theHits[idOne].delta = i_delta;
        theHits[idOne].nearestHigher = i_nearestHigher;
      } else {
        // otherwise delta is garanteed to be larger outlierDeltaFactor_*delta_c
        // we can safely maximize delta to be maxDelta
        theHits[idOne].delta = maxDelta;
        theHits[idOne].nearestHigher = -1;
      }
    }
  } //kernel


  __global__ void kernel_find_clusters( RecHitGPU* theHits, GPU::VecArray<int,maxNSeeds>* seeds,
                                        float* vecDeltas_,//float delta_c, 
                                        int theHitsSize, 
                                        float kappa_, 
                                        float outlierDeltaFactor_
                                        ) {

    size_t idOne = blockIdx.x * blockDim.x + threadIdx.x;
    size_t layer = theHits[idOne].layer;

    float delta_c;

    if (layer <= lastLayerEE)
      delta_c = vecDeltas_[0];
    else if (layer <= lastLayerFH)
      delta_c = vecDeltas_[1];
    else
      delta_c = vecDeltas_[2];
    
    if (idOne < theHitsSize){

      float rho_c = kappa_ * theHits[idOne].sigmaNoise;

      // initialize clusterIndex
      theHits[idOne].clusterIndex = -1;

      bool isSeed = (theHits[idOne].delta > delta_c) && (theHits[idOne].rho >= rho_c);
      bool isOutlier = (theHits[idOne].delta > outlierDeltaFactor_*delta_c) && (theHits[idOne].rho < rho_c);

      if (isSeed) {
        // decide a seed
        // push hits[idOne] into seeds
        seeds[0].push_back(idOne);
      } else {
        if (!isOutlier) {
          // if not a seed or outlier
          // register yourself as follower of your nearest higher
          int idNH = theHits[idOne].nearestHigher;
          theHits[idNH].followers.push_back(idOne);  
        }
      }
    }
  } //kernel


  __global__ void kernel_assign_clusters( RecHitGPU* theHits, GPU::VecArray<int,maxNSeeds>* seeds ) {

    unsigned int idOne = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nClusters = seeds[0].size();

    if (idOne < nClusters){


      int buffer[BufferSizePerSeed];
      int bufferSize = 0;

      // asgine cluster to seed[idOne]
      int idThisSeed = seeds[0][idOne];
      theHits[idThisSeed].clusterIndex = idOne;
      // push_back idThisSeed to buffer
      buffer[bufferSize] = idThisSeed;
      bufferSize ++;

      // process all elements in buffer
      while (bufferSize>0){
        // get last element of buffer
        int idEndOfBuffer = buffer[bufferSize-1];
        RecHitGPU thisHit = theHits[ idEndOfBuffer ];
                
        // pop_back last element of buffer
        buffer[bufferSize-1] = 0;
        bufferSize--;

        // loop over followers of last element of buffer
        for( int j=0; j < thisHit.followers.size();j++ ){
          // pass id to follower
          theHits[thisHit.followers[j]].clusterIndex = thisHit.clusterIndex;
          
          // push_back follower to buffer
          buffer[bufferSize] = thisHit.followers[j];
          bufferSize++;
        }
      }

    }
  } //kernel





  double clue_BinGPU( HgcRecHitsGPU& theHits,
                      std::vector<double> vecDeltas_, 
                      float kappa_, 
                      float outlierDeltaFactor_) {

    double maxdensity = 0.;
    
    double* vecDeltasData = vecDeltas_.data();
    float vecDeltasFloat[3];
    std::copy(vecDeltasData, vecDeltasData+3, vecDeltasFloat);
    
    // find total number of hits
    unsigned int totalNumberOfHits = 0;
    for(unsigned int layer = 0; layer < theHits.size(); layer++){
      totalNumberOfHits += theHits[layer].size();
    }
    // make input hits for GPU, allocating enough memory
    RecHitGPU *hInputRecHits = new RecHitGPU[totalNumberOfHits];
    RecHitGPU *dInputRecHits;
    
    // copy 2D vector theHits into 1D array hInputRecHits
    unsigned int nHits = 0;
    for(unsigned int layer = 0; layer < theHits.size(); layer++){
      std::copy(theHits[layer].begin(), theHits[layer].end(), hInputRecHits+nHits);
      nHits += theHits[layer].size();
    }
    cudaMalloc(&dInputRecHits, sizeof(RecHitGPU)*totalNumberOfHits);
    cudaMemcpy(dInputRecHits, hInputRecHits, sizeof(RecHitGPU)*totalNumberOfHits, cudaMemcpyHostToDevice);
    
    
    // histogram
    float minX = -250.0, minY = -250.0;
    float maxX =  250.0, maxY =  250.0;
    Histo2D hHist(minX,maxX,minY,maxY);
    Histo2D *dHist;
    cudaMalloc(&dHist, sizeof(Histo2D)*theHits.size());
    int sizeLayer = 0;
    for(unsigned int i = 0; i < theHits.size(); i++ ){
      cudaMemcpy(dHist+sizeLayer, &hHist, sizeof(Histo2D), cudaMemcpyHostToDevice);
      sizeLayer += hHist.size();
    }
    

    // define dSeeds as GPU vecArray
    GPU::VecArray<int,maxNSeeds> *dSeeds;
    cudaMalloc(&dSeeds, sizeof(GPU::VecArray<int,maxNSeeds>));
    cudaMemset(dSeeds, 0x00, sizeof(GPU::VecArray<int,maxNSeeds>));
    
    
    // Call the kernel
    const dim3 blockSize(1024,1,1);
    const dim3 gridSize(ceil(totalNumberOfHits/1024.),1,1);

    kernel_compute_histogram <<<gridSize,blockSize>>>(dHist, dInputRecHits,  totalNumberOfHits);
    cudaDeviceSynchronize();

    cudaMemcpy(&hHist, dHist, sizeof(Histo2D), cudaMemcpyDeviceToHost);
    for(unsigned int i = 0; i < theHits.size(); i++){
      std::cout << "Size of layer " << i << ": " << hHist[i].size() << std::endl;
    }
    
    kernel_compute_density <<<gridSize,blockSize>>>(dHist, dInputRecHits, 
                                                    vecDeltasFloat,//delta_c, 
                                                    totalNumberOfHits//theHits.data().size()
                                                    );

    kernel_compute_distanceToHigher <<<gridSize,blockSize>>>( dHist, dInputRecHits, 
                                                              vecDeltasFloat,//delta_c, 
                                                              totalNumberOfHits, 
                                                              outlierDeltaFactor_
                                                              );

    // kernel_find_clusters <<<gridSize,blockSize>>>(dInputRecHits, dSeeds,
    //                                               vecDeltasFloat,//delta_c, 
    //                                               totalNumberOfHits, 
    //                                               kappa_, 
    //                                               outlierDeltaFactor_
    //                                               );

    //const dim3 blockSize_assign_clusters(maxNSeeds,1,1);                                              
    //const dim3 gridSize_assign_clusters(1,1,1);
    // kernel_assign_clusters <<<gridSize_assign_clusters,blockSize_assign_clusters>>>(dInputRecHits, dSeeds);

    // Copy result back!/
    cudaMemcpy(hInputRecHits, dInputRecHits, sizeof(RecHitGPU)*totalNumberOfHits, cudaMemcpyDeviceToHost);


    // Free all the memory
    cudaFree(dInputRecHits);
    cudaFree(dHist);
    cudaFree(dSeeds);
    
    // for(unsigned int j = 0; j< totalNumberOfHits; j++) {
    //   if (maxdensity < theHits[j].rho) {
    //     maxdensity = theHits[j].rho;
    //   }
    // } UNCOMMENT LATER

    return maxdensity;

  }




}//namespace