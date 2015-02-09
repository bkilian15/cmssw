import FWCore.ParameterSet.Config as cms

# step 0.5

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeLowPtTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeLowPtTripletSeeds.simTrackSelection.skipSimTrackIdTags = [
    cms.InputTag("initialStepSimTrackIds"),
    cms.InputTag("detachedTripletStepSimTrackIds")]
iterativeLowPtTripletSeeds.simTrackSelection.minLayersCrossed = 3
iterativeLowPtTripletSeeds.simTrackSelection.pTMin = 0.25
iterativeLowPtTripletSeeds.simTrackSelection.maxD0 = 5.
iterativeLowPtTripletSeeds.simTrackSelection.maxZ0 = 50.
iterativeLowPtTripletSeeds.outputSeedCollectionName = 'LowPtPixelTriplets'
iterativeLowPtTripletSeeds.originRadius = 0.03
iterativeLowPtTripletSeeds.originHalfLength = 17.5
iterativeLowPtTripletSeeds.originpTMin = 0.35
iterativeLowPtTripletSeeds.primaryVertex = ''

#iterativeLowPtTripletSeeds.layerList = ['BPix1+BPix2+BPix3',
#                                   'BPix1+BPix2+FPix1_pos',
#                                   'BPix1+BPix2+FPix1_neg',
#                                   'BPix1+FPix1_pos+FPix2_pos',
#                                   'BPix1+FPix1_neg+FPix2_neg']
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeLowPtTripletSeeds.layerList = PixelLayerTriplets.layerList

# candidate producer

from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
lowPtTripletStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativeLowPtTripletSeeds","LowPtPixelTriplets"),
    MinNumberOfCrossedLayers = 3)

# track producer

from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStepTracks
lowPtTripletStepTracks = lowPtTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    Propagator = 'PropagatorWithMaterial',
    TTRHBuilder = 'WithoutRefit'
)

# simtrack id producer
lowPtTripletStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                     trackCollection = cms.InputTag("lowPtTripletStepTracks"),
                                     HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                     )

# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStepSelector
lowPtTripletStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

LowPtTripletStep = cms.Sequence(iterativeLowPtTripletSeeds+
                                lowPtTripletStepTrackCandidates+
                                lowPtTripletStepTracks+  
                                lowPtTripletStepSimTrackIds+
                                lowPtTripletStepSelector)

