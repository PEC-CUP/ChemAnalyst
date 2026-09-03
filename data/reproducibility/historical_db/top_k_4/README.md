# petroleum fraction Historical Experimental DB evaluation

This run evaluates training-free property inference using train/test sample workbooks.

## Data split
- Train samples: 54
- Test samples: 14

## Feature processing
- GC and IR curves are aligned to a training-set grid and converted into PCA/PLS latent variables.
- PCA models are fitted per spectral channel using train samples only.
- PLS models are fitted per spectral channel and target property using train samples only.
- Hydrocarbon-type rows are converted into component-content features.
- Bulk properties are numeric features only when the scenario allows them; the target property itself is always masked from the query.

## Metrics
| target_property | scenario | method | n | MAE | RMSE | MRE_percent | MAPE_percent | bias | median_abs_error | range_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BP_20 | analysis_only | correlation_weighted | 14 | 29.082218 | 38.979065 | 0.480576 | 8.027433 | -7.664476 | 16.469427 | 0.428571 |
| BP_20 | analysis_only | equal_weight | 14 | 33.025774 | 43.432616 | 0.417303 | 8.953272 | -8.732419 | 19.332792 | 0.357143 |
| BP_20 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 28.67913 | 37.120938 | 1.39648 | 8.11989 | -4.296606 | 20.92426 | 0.357143 |
| BP_20 | analysis_plus_non_target_bulk | equal_weight | 14 | 29.890567 | 39.492392 | 0.768474 | 8.269268 | -6.756737 | 16.723696 | 0.357143 |
| BP_35 | analysis_only | correlation_weighted | 14 | 22.454493 | 32.015747 | -0.084094 | 5.85841 | -6.852552 | 10.89303 | 0.571429 |
| BP_35 | analysis_only | equal_weight | 14 | 26.745112 | 37.1839 | -0.254796 | 6.817573 | -8.314483 | 13.350346 | 0.642857 |
| BP_35 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 20.543359 | 30.517491 | 0.529028 | 5.425381 | -4.450923 | 8.131503 | 0.642857 |
| BP_35 | analysis_plus_non_target_bulk | equal_weight | 14 | 23.296843 | 33.966576 | 0.017405 | 6.075793 | -6.64277 | 11.170119 | 0.642857 |
| BP_50 | analysis_only | correlation_weighted | 14 | 19.081151 | 28.249659 | -0.519698 | 5.025872 | -7.13014 | 7.248166 | 0.642857 |
| BP_50 | analysis_only | equal_weight | 14 | 23.841667 | 33.56685 | -0.541732 | 6.046257 | -7.895498 | 15.22146 | 0.642857 |
| BP_50 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 16.902749 | 26.608581 | 0.046983 | 4.500493 | -4.762772 | 5.455562 | 0.714286 |
| BP_50 | analysis_plus_non_target_bulk | equal_weight | 14 | 20.703953 | 31.60655 | -0.30901 | 5.408246 | -6.431714 | 7.387723 | 0.642857 |
| BP_65 | analysis_only | correlation_weighted | 14 | 21.126625 | 29.747381 | -0.775084 | 5.410651 | -8.040334 | 11.664649 | 0.714286 |
| BP_65 | analysis_only | equal_weight | 14 | 23.730064 | 34.821938 | -0.889326 | 5.911814 | -9.168136 | 10.437037 | 0.714286 |
| BP_65 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 19.546805 | 28.206306 | -0.394037 | 5.028537 | -6.423848 | 10.557703 | 0.785714 |
| BP_65 | analysis_plus_non_target_bulk | equal_weight | 14 | 23.29625 | 34.186621 | -0.69904 | 5.862108 | -7.949261 | 12.140079 | 0.714286 |
| BP_80 | analysis_only | correlation_weighted | 14 | 28.826892 | 38.178445 | -1.079862 | 6.624039 | -10.417419 | 19.520815 | 0.714286 |
| BP_80 | analysis_only | equal_weight | 14 | 28.918056 | 41.819554 | -1.441637 | 6.633755 | -12.461967 | 11.858914 | 0.642857 |
| BP_80 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 26.638169 | 36.425848 | -0.995866 | 6.143068 | -9.928567 | 17.541003 | 0.714286 |
| BP_80 | analysis_plus_non_target_bulk | equal_weight | 14 | 30.600058 | 42.558526 | -1.248953 | 6.970362 | -11.269856 | 19.690219 | 0.642857 |
| density_20c | analysis_only | correlation_weighted | 14 | 0.020605 | 0.033278 | -0.648448 | 2.203289 | -0.007566 | 0.00897 | 0.857143 |
| density_20c | analysis_only | equal_weight | 14 | 0.021577 | 0.033664 | -0.680713 | 2.311662 | -0.007867 | 0.010639 | 0.857143 |
| density_20c | analysis_plus_non_target_bulk | correlation_weighted | 14 | 0.018794 | 0.027344 | -0.506872 | 2.051354 | -0.00585 | 0.009007 | 0.785714 |
| density_20c | analysis_plus_non_target_bulk | equal_weight | 14 | 0.020659 | 0.029375 | -0.310724 | 2.249804 | -0.004244 | 0.010842 | 0.785714 |
| saturates | analysis_only | correlation_weighted | 14 | 8.514151 | 12.650945 | 16.165221 | 20.658543 | 4.525671 | 5.267896 | 0.642857 |
| saturates | analysis_only | equal_weight | 14 | 7.761747 | 11.465109 | 13.997461 | 19.118869 | 3.241667 | 5.890339 | 0.785714 |
| saturates | analysis_plus_non_target_bulk | correlation_weighted | 14 | 6.407976 | 8.030387 | 8.185216 | 13.398765 | 1.881942 | 5.221699 | 0.785714 |
| saturates | analysis_plus_non_target_bulk | equal_weight | 14 | 7.217505 | 9.3313 | 9.776948 | 15.693713 | 2.055397 | 6.016005 | 0.785714 |

## Top feature-target relationships
### BP_20
| feature | feature_channel | n | spearman | pearson | weight |
| --- | --- | --- | --- | --- | --- |
| bulk_property:bp_35 | bulk_property | 54 | 0.988277 | 0.991321 | 1.039799 |
| bulk_property:bp_50 | bulk_property | 54 | 0.960984 | 0.969274 | 1.015129 |
| bulk_property:bp_65 | bulk_property | 54 | 0.932088 | 0.939426 | 0.985757 |
| gc_fid:pls:density_20c:lv1 | gc_fid | 54 | 0.898927 | 0.897661 | 0.948294 |
| gc_fid:pls:bp_20:lv1 | gc_fid | 54 | 0.900147 | 0.87996 | 0.940054 |
| gc_fid:pls:bp_35:lv1 | gc_fid | 54 | 0.896983 | 0.881694 | 0.939338 |
| bulk_property:bp_80 | bulk_property | 54 | 0.88955 | 0.888065 | 0.938808 |
| gc_fid:pls:bp_50:lv1 | gc_fid | 54 | 0.89359 | 0.881437 | 0.937514 |
| gc_fid:pls:bp_65:lv1 | gc_fid | 54 | 0.892256 | 0.880348 | 0.936302 |
| gc_fid:pls:bp_80:lv1 | gc_fid | 54 | 0.890274 | 0.878036 | 0.934155 |

### BP_35
| feature | feature_channel | n | spearman | pearson | weight |
| --- | --- | --- | --- | --- | --- |
| bulk_property:bp_20 | bulk_property | 54 | 0.988277 | 0.991321 | 1.039799 |
| bulk_property:bp_50 | bulk_property | 54 | 0.987248 | 0.991883 | 1.039566 |
| bulk_property:bp_65 | bulk_property | 54 | 0.966642 | 0.9726 | 1.019621 |
| bulk_property:bp_80 | bulk_property | 54 | 0.933115 | 0.931783 | 0.982449 |
| gc_fid:pls:density_20c:lv1 | gc_fid | 54 | 0.9119 | 0.9224 | 0.96715 |
| gc_fid:pls:bp_35:lv1 | gc_fid | 54 | 0.905305 | 0.892533 | 0.948919 |
| gc_fid:pls:saturates:lv1 | gc_fid | 54 | -0.89444 | -0.90234 | 0.94839 |
| gc_fid:pls:bp_20:lv1 | gc_fid | 54 | 0.907516 | 0.889124 | 0.94832 |
| gc_fid:pls:bp_50:lv1 | gc_fid | 54 | 0.902217 | 0.893375 | 0.947796 |
| gc_fid:pls:bp_65:lv1 | gc_fid | 54 | 0.900844 | 0.892959 | 0.946902 |

### BP_50
| feature | feature_channel | n | spearman | pearson | weight |
| --- | --- | --- | --- | --- | --- |
| bulk_property:bp_65 | bulk_property | 54 | 0.991175 | 0.993547 | 1.042361 |
| bulk_property:bp_35 | bulk_property | 54 | 0.987248 | 0.991883 | 1.039566 |
| bulk_property:bp_80 | bulk_property | 54 | 0.967618 | 0.967513 | 1.017565 |
| bulk_property:bp_20 | bulk_property | 54 | 0.960984 | 0.969274 | 1.015129 |
| gc_fid:pls:density_20c:lv1 | gc_fid | 54 | 0.912612 | 0.940463 | 0.976537 |
| gc_fid:pls:saturates:lv1 | gc_fid | 54 | -0.89542 | -0.924638 | 0.960029 |
| gc_fid:pls:bp_50:lv1 | gc_fid | 54 | 0.898165 | 0.903734 | 0.950949 |
| gc_fid:pls:bp_80:lv1 | gc_fid | 54 | 0.898088 | 0.903499 | 0.950794 |
| gc_fid:pls:bp_35:lv1 | gc_fid | 54 | 0.899689 | 0.901859 | 0.950774 |
| gc_fid:pls:bp_65:lv1 | gc_fid | 54 | 0.897021 | 0.904189 | 0.950605 |

### BP_65
| feature | feature_channel | n | spearman | pearson | weight |
| --- | --- | --- | --- | --- | --- |
| bulk_property:bp_50 | bulk_property | 54 | 0.991175 | 0.993547 | 1.042361 |
| bulk_property:bp_80 | bulk_property | 54 | 0.989231 | 0.989434 | 1.039333 |
| bulk_property:bp_35 | bulk_property | 54 | 0.966642 | 0.9726 | 1.019621 |
| bulk_property:bp_20 | bulk_property | 54 | 0.932088 | 0.939426 | 0.985757 |
| gc_fid:pls:density_20c:lv1 | gc_fid | 54 | 0.916098 | 0.943845 | 0.979972 |
| gc_fid:pls:saturates:lv1 | gc_fid | 54 | -0.901003 | -0.929367 | 0.965185 |
| gc_fid:pls:bp_80:lv1 | gc_fid | 54 | 0.900964 | 0.904009 | 0.952487 |
| gc_fid:pls:bp_65:lv1 | gc_fid | 54 | 0.899135 | 0.903458 | 0.951297 |
| gc_fid:pls:bp_50:lv1 | gc_fid | 54 | 0.90005 | 0.902007 | 0.951029 |
| gc_fid:pls:bp_35:lv1 | gc_fid | 54 | 0.900583 | 0.899264 | 0.949924 |

### BP_80
| feature | feature_channel | n | spearman | pearson | weight |
| --- | --- | --- | --- | --- | --- |
| bulk_property:bp_65 | bulk_property | 54 | 0.989231 | 0.989434 | 1.039333 |
| bulk_property:bp_50 | bulk_property | 54 | 0.967618 | 0.967513 | 1.017565 |
| bulk_property:bp_35 | bulk_property | 54 | 0.933115 | 0.931783 | 0.982449 |
| gc_fid:pls:density_20c:lv1 | gc_fid | 54 | 0.908992 | 0.937368 | 0.97318 |
| gc_fid:pls:saturates:lv1 | gc_fid | 54 | -0.898357 | -0.925239 | 0.961798 |
| gc_fid:pls:bp_80:lv1 | gc_fid | 54 | 0.893059 | 0.896234 | 0.944646 |
| gc_fid:pls:bp_65:lv1 | gc_fid | 54 | 0.890047 | 0.894084 | 0.942065 |
| gc_fid:pls:bp_50:lv1 | gc_fid | 54 | 0.890657 | 0.891423 | 0.94104 |
| gc_fid:pls:bp_35:lv1 | gc_fid | 54 | 0.8902 | 0.887698 | 0.938949 |
| bulk_property:bp_20 | bulk_property | 54 | 0.88955 | 0.888065 | 0.938808 |

### density_20c
| feature | feature_channel | n | spearman | pearson | weight |
| --- | --- | --- | --- | --- | --- |
| bulk_property:saturates | bulk_property | 54 | -0.817771 | -0.873225 | 0.895498 |
| ir:weighted_x_q90 | ir | 52 | 0.681653 | 0.727177 | 0.728325 |
| ir_spectrum:pls:density_20c:lv2 | ir_spectrum | 52 | 0.742086 | 0.664839 | 0.727409 |
| ir_spectrum:pls:saturates:lv2 | ir_spectrum | 52 | -0.684839 | -0.698491 | 0.716048 |
| ir_spectrum:pls:density_20c:lv1 | ir_spectrum | 52 | 0.616278 | 0.627487 | 0.64885 |
| ir_spectrum:pls:saturates:lv1 | ir_spectrum | 52 | -0.602147 | -0.601065 | 0.629324 |
| ir_spectrum:pls:bp_35:lv1 | ir_spectrum | 52 | 0.591944 | 0.609068 | 0.628265 |
| ir_spectrum:pca:pc1 | ir_spectrum | 52 | -0.600013 | -0.573981 | 0.615256 |
| ir_spectrum:pls:bp_65:lv1 | ir_spectrum | 52 | 0.566586 | 0.606183 | 0.614667 |
| ir_spectrum:pls:bp_50:lv1 | ir_spectrum | 52 | 0.571069 | 0.601573 | 0.614605 |

### saturates
| feature | feature_channel | n | spearman | pearson | weight |
| --- | --- | --- | --- | --- | --- |
| bulk_property:density_20c | bulk_property | 54 | -0.817771 | -0.873225 | 0.895498 |
| ir:weighted_x_q90 | ir | 52 | -0.657977 | -0.697264 | 0.702523 |
| ir_spectrum:pls:density_20c:lv1 | ir_spectrum | 52 | -0.620422 | -0.663573 | 0.668219 |
| ir_spectrum:pls:saturates:lv1 | ir_spectrum | 52 | 0.627508 | 0.656323 | 0.668141 |
| ir_spectrum:pca:pc1 | ir_spectrum | 52 | 0.622898 | 0.635169 | 0.655736 |
| ir:weighted_x_mean | ir | 52 | -0.623922 | -0.544614 | 0.612628 |
| ir_spectrum:pls:bp_80:lv1 | ir_spectrum | 52 | -0.465295 | -0.629123 | 0.576942 |
| ir_spectrum:pls:saturates:lv2 | ir_spectrum | 52 | 0.52907 | 0.545746 | 0.567504 |
| ir:weighted_x_q50 | ir | 52 | -0.613131 | -0.459753 | 0.566574 |
| ir_spectrum:pls:bp_65:lv1 | ir_spectrum | 52 | -0.432169 | -0.59105 | 0.542661 |

## Prediction examples
| sample_id | raw_sample_id | target_property | scenario | method | actual | estimate | error | abs_error | relative_error_percent | absolute_percentage_error | neighbor_count | nearest_distance | historical_range_min | historical_range_max | actual_in_neighbor_range | mean_shared_feature_count | top_neighbor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_only | equal_weight | 0.8713 | 0.861704 | -0.009596 | 0.009596 | -1.101377 | 1.101377 | 4 | 0.308593 | 0.8348 | 0.9004 | True | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_only | correlation_weighted | 0.8713 | 0.869152 | -0.002148 | 0.002148 | -0.246493 | 0.246493 | 4 | 0.345293 | 0.8498 | 0.9004 | True | 74.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_plus_non_target_bulk | equal_weight | 0.8713 | 0.860621 | -0.010679 | 0.010679 | -1.225612 | 1.225612 | 4 | 0.29803 | 0.8348 | 0.9004 | True | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_plus_non_target_bulk | correlation_weighted | 0.8713 | 0.854733 | -0.016567 | 0.016567 | -1.901377 | 1.901377 | 4 | 0.328162 | 0.8348 | 0.8712 | False | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_only | equal_weight | 91.31 | 85.270636 | -6.039364 | 6.039364 | -6.614133 | 6.614133 | 4 | 0.308593 | 71.554633 | 96.271711 | True | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_only | correlation_weighted | 91.31 | 85.378148 | -5.931852 | 5.931852 | -6.496389 | 6.496389 | 4 | 0.354559 | 71.554633 | 96.271711 | True | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_plus_non_target_bulk | equal_weight | 91.31 | 85.51036 | -5.79964 | 5.79964 | -6.351594 | 6.351594 | 4 | 0.302101 | 71.554633 | 96.271711 | True | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_plus_non_target_bulk | correlation_weighted | 91.31 | 89.427715 | -1.882285 | 1.882285 | -2.061423 | 2.061423 | 4 | 0.347395 | 84.039121 | 96.271711 | True | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_only | equal_weight | 350.6 | 363.241511 | 12.641511 | 12.641511 | 3.605679 | 3.605679 | 4 | 0.308593 | 351.2 | 378.4 | False | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_only | correlation_weighted | 350.6 | 362.792668 | 12.192668 | 12.192668 | 3.477658 | 3.477658 | 4 | 0.295591 | 351.2 | 378.4 | False | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_plus_non_target_bulk | equal_weight | 350.6 | 362.872312 | 12.272312 | 12.272312 | 3.500374 | 3.500374 | 4 | 0.302348 | 351.2 | 378.4 | False | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_plus_non_target_bulk | correlation_weighted | 350.6 | 359.908787 | 9.308787 | 9.308787 | 2.655102 | 2.655102 | 4 | 0.279582 | 351.2 | 366.4 | False | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_only | equal_weight | 378.2 | 391.351389 | 13.151389 | 13.151389 | 3.477363 | 3.477363 | 4 | 0.308593 | 378.0 | 411.2 | True | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_only | correlation_weighted | 378.2 | 390.74349 | 12.54349 | 12.54349 | 3.316629 | 3.316629 | 4 | 0.294253 | 378.0 | 411.2 | True | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_plus_non_target_bulk | equal_weight | 378.2 | 390.882213 | 12.682213 | 12.682213 | 3.353309 | 3.353309 | 4 | 0.302753 | 378.0 | 411.2 | True | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_plus_non_target_bulk | correlation_weighted | 378.2 | 384.375295 | 6.175295 | 6.175295 | 1.632812 | 1.632812 | 4 | 0.279407 | 378.0 | 400.2 | True | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_only | equal_weight | 399.2 | 414.274696 | 15.074696 | 15.074696 | 3.776227 | 3.776227 | 4 | 0.308593 | 395.6 | 438.0 | True | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_only | correlation_weighted | 399.2 | 404.987308 | 5.787308 | 5.787308 | 1.449726 | 1.449726 | 4 | 0.292267 | 395.6 | 428.0 | True | 63.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_plus_non_target_bulk | equal_weight | 399.2 | 413.720044 | 14.520044 | 14.520044 | 3.637286 | 3.637286 | 4 | 0.302672 | 395.6 | 438.0 | True | 69.75 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_plus_non_target_bulk | correlation_weighted | 399.2 | 404.67889 | 5.47889 | 5.47889 | 1.372468 | 1.372468 | 4 | 0.277295 | 395.6 | 428.0 | True | 69.75 | petroleum_fraction_db_009 |
