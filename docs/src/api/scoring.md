# Scoring API

```@meta
CurrentModule = RobustVisualGeometry
```

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
ConvergeThenRescore
StepAndRescore
```

## Covariance Structure

```@docs
CovarianceStructure
Homoscedastic
Heteroscedastic
Predictive
covariance_structure
```
