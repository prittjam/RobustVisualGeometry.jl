# Scoring API

## Quality Functions

```@docs
AbstractQualityFunction
AbstractMarginalQuality
MarginalQuality
PredictiveMarginalQuality
init_quality
default_local_optimization
```

## Local Optimization

```@docs
AbstractLocalOptimization
NoLocalOptimization
```

## Stopping Strategies

```@docs
AbstractStoppingStrategy
HypergeometricStopping
ScoreGapStopping
stopping_strategy
```
