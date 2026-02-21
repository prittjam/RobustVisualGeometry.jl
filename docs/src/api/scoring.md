# Scoring API

## Quality Functions

```@docs
AbstractQualityFunction
ThresholdQuality
ChiSquareQuality
TruncatedQuality
AbstractMarginalQuality
MarginalQuality
PredictiveMarginalQuality
init_quality
default_local_optimization
```

## F-Test Types

```@docs
AbstractTestType
BasicFTest
PredictiveFTest
test_type
```

## Local Optimization

```@docs
AbstractLocalOptimization
NoLocalOptimization
SimpleRefit
FTestLocalOptimization
```

## Stopping Strategies

```@docs
AbstractStoppingStrategy
HypergeometricStopping
ScoreGapStopping
stopping_strategy
```
