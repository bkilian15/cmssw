import FWCore.ParameterSet.Config as cms

# step 1

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelPairSeeds.simTrackSelection.skipSimTrackIdTags = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds")]
iterativePixelPairSeeds.simTrackSelection.skipSimTrackIdTags = [cms.InputTag("initialStepIds"), cms.InputTag("lowPtTripletStepIds")]
iterativePixelPairSeeds.simTrackSelection.minLayersCrossed =3
iterativePixelPairSeeds.simTrackSelection.pTMin = 0.3
iterativePixelPairSeeds.simTrackSelection.maxD0 = 5.
iterativePixelPairSeeds.simTrackSelection.maxZ0 = 50.
iterativePixelPairSeeds.outputSeedCollectionName = 'PixelPair'
iterativePixelPairSeeds.originRadius = 0.2
iterativePixelPairSeeds.originHalfLength = 17.5
iterativePixelPairSeeds.originpTMin = 0.6

iterativePixelPairSeeds.beamSpot = ''
iterativePixelPairSeeds.primaryVertex = 'firstStepPrimaryVertices' # vertices are generated from the initalStepTracks

#iterativePixelPairSeeds.layerList = ['BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
#                                     'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
#                                     'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 
#                                     'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
#                                     'BPix2+FPix2_pos', 'BPix2+FPix2_neg', 
#                                     'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg', 
#                                     'FPix2_pos+TEC1_pos', 'FPix2_pos+TEC2_pos', 
#                                     'FPix2_neg+TEC1_neg', 'FPix2_neg+TEC2_neg']
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSeedLayers
iterativePixelPairSeeds.layerList = pixelPairStepSeedLayers.layerList

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
pixelPairStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativePixelPairSeeds","PixelPair"),
    MinNumberOfCrossedLayers = 2 # ?
)

# track producer
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepTracks
pixelPairStepTracks = pixelPairStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherSecond',
    Propagator = 'PropagatorWithMaterial',
)

# simtrack id producer
pixelPairStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                          trackCollection = cms.InputTag("pixelPairStepTracks"),
                                          HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                          )

# Final selection
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSelector
pixelPairStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# sequence
PixelPairStep = cms.Sequence(iterativePixelPairSeeds+
                             pixelPairStepTrackCandidates+
                             pixelPairStepTracks+
                             pixelPairStepSimTrackIds+
                             pixelPairStepSelector)
