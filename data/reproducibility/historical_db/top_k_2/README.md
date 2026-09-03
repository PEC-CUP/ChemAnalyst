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
| BP_20 | analysis_only | correlation_weighted | 14 | 28.630861 | 38.611604 | 0.463323 | 7.747334 | -6.863116 | 19.374634 | 0.071429 |
| BP_20 | analysis_only | equal_weight | 14 | 27.758488 | 34.48734 | 0.933414 | 7.707459 | -4.767811 | 20.015491 | 0.214286 |
| BP_20 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 27.942771 | 38.549084 | 0.20961 | 7.533501 | -7.801749 | 19.356954 | 0.214286 |
| BP_20 | analysis_plus_non_target_bulk | equal_weight | 14 | 27.218042 | 34.03883 | 0.84517 | 7.519373 | -4.867189 | 19.697599 | 0.214286 |
| BP_35 | analysis_only | correlation_weighted | 14 | 24.635699 | 32.847279 | -0.265193 | 6.422906 | -7.207008 | 16.232448 | 0.142857 |
| BP_35 | analysis_only | equal_weight | 14 | 21.528417 | 28.418484 | 0.787925 | 5.874047 | -2.470203 | 14.948151 | 0.357143 |
| BP_35 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 22.439687 | 31.782922 | -0.089165 | 5.94293 | -6.368448 | 14.468388 | 0.285714 |
| BP_35 | analysis_plus_non_target_bulk | equal_weight | 14 | 21.065002 | 28.025122 | 0.629509 | 5.748203 | -3.07307 | 14.743616 | 0.357143 |
| BP_50 | analysis_only | correlation_weighted | 14 | 19.623171 | 28.190624 | -0.129227 | 5.271602 | -5.237339 | 10.934452 | 0.428571 |
| BP_50 | analysis_only | equal_weight | 14 | 20.622451 | 27.35794 | 0.888819 | 5.57514 | -0.488916 | 13.236739 | 0.357143 |
| BP_50 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 19.027525 | 28.010859 | -0.121052 | 5.169073 | -5.076877 | 11.147959 | 0.357143 |
| BP_50 | analysis_plus_non_target_bulk | equal_weight | 14 | 19.839416 | 26.900667 | 0.604956 | 5.385327 | -1.699789 | 12.900225 | 0.428571 |
| BP_65 | analysis_only | correlation_weighted | 14 | 21.210262 | 31.189953 | 0.413898 | 5.526311 | -2.429337 | 9.694245 | 0.428571 |
| BP_65 | analysis_only | equal_weight | 14 | 21.49963 | 30.896218 | 0.910903 | 5.645506 | -0.086263 | 11.118887 | 0.428571 |
| BP_65 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 20.016032 | 30.408009 | -0.442077 | 5.347419 | -6.388177 | 8.887859 | 0.428571 |
| BP_65 | analysis_plus_non_target_bulk | equal_weight | 14 | 21.250245 | 30.823698 | 0.498106 | 5.584289 | -2.006953 | 10.806676 | 0.5 |
| BP_80 | analysis_only | correlation_weighted | 14 | 29.126306 | 39.768556 | 0.437851 | 6.809393 | -2.905031 | 13.549176 | 0.428571 |
| BP_80 | analysis_only | equal_weight | 14 | 29.39202 | 39.297391 | 0.87056 | 6.9429 | -0.661814 | 17.821521 | 0.428571 |
| BP_80 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 31.453958 | 40.899503 | -0.442534 | 7.337383 | -6.927915 | 21.757031 | 0.357143 |
| BP_80 | analysis_plus_non_target_bulk | equal_weight | 14 | 29.808305 | 39.98331 | 0.371827 | 6.99632 | -3.175278 | 17.733028 | 0.5 |
| density_20c | analysis_only | correlation_weighted | 14 | 0.021824 | 0.033065 | -0.608909 | 2.397094 | -0.006805 | 0.014979 | 0.357143 |
| density_20c | analysis_only | equal_weight | 14 | 0.021494 | 0.031298 | -0.594796 | 2.348192 | -0.006592 | 0.014756 | 0.357143 |
| density_20c | analysis_plus_non_target_bulk | correlation_weighted | 14 | 0.018422 | 0.026666 | 0.019315 | 2.070884 | -0.000696 | 0.012868 | 0.285714 |
| density_20c | analysis_plus_non_target_bulk | equal_weight | 14 | 0.020192 | 0.031517 | -0.165498 | 2.265988 | -0.002387 | 0.014177 | 0.285714 |
| saturates | analysis_only | correlation_weighted | 14 | 6.540257 | 9.085022 | 9.054351 | 13.370039 | 2.702002 | 3.782415 | 0.428571 |
| saturates | analysis_only | equal_weight | 14 | 6.585511 | 8.720532 | 6.963751 | 12.836831 | 1.447427 | 4.605161 | 0.357143 |
| saturates | analysis_plus_non_target_bulk | correlation_weighted | 14 | 4.965407 | 6.127049 | 1.595842 | 7.3446 | -0.073652 | 3.76247 | 0.357143 |
| saturates | analysis_plus_non_target_bulk | equal_weight | 14 | 5.477773 | 7.387839 | 2.061712 | 7.796232 | 0.440074 | 3.985854 | 0.357143 |

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
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_only | equal_weight | 0.8713 | 0.856427 | -0.014873 | 0.014873 | -1.706981 | 1.706981 | 2 | 0.308593 | 0.8498 | 0.8653 | False | 74.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_only | correlation_weighted | 0.8713 | 0.856739 | -0.014561 | 0.014561 | -1.671146 | 1.671146 | 2 | 0.345293 | 0.8498 | 0.8653 | False | 74.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_plus_non_target_bulk | equal_weight | 0.8713 | 0.856252 | -0.015048 | 0.015048 | -1.727046 | 1.727046 | 2 | 0.29803 | 0.8498 | 0.8653 | False | 80.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_plus_non_target_bulk | correlation_weighted | 0.8713 | 0.856489 | -0.014811 | 0.014811 | -1.699891 | 1.699891 | 2 | 0.328162 | 0.8498 | 0.8653 | False | 80.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_only | equal_weight | 91.31 | 86.67791 | -4.63209 | 4.63209 | -5.072927 | 5.072927 | 2 | 0.308593 | 84.039121 | 88.648789 | False | 74.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_only | correlation_weighted | 91.31 | 86.532854 | -4.777146 | 4.777146 | -5.231789 | 5.231789 | 2 | 0.354559 | 84.039121 | 88.648789 | False | 74.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_plus_non_target_bulk | equal_weight | 91.31 | 86.70597 | -4.60403 | 4.60403 | -5.042197 | 5.042197 | 2 | 0.302101 | 84.039121 | 88.648789 | False | 80.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_plus_non_target_bulk | correlation_weighted | 91.31 | 86.553836 | -4.756164 | 4.756164 | -5.208809 | 5.208809 | 2 | 0.347395 | 84.039121 | 88.648789 | False | 80.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_only | equal_weight | 350.6 | 361.649405 | 11.049405 | 11.049405 | 3.15157 | 3.15157 | 2 | 0.308593 | 358.4 | 366.0 | False | 74.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_only | correlation_weighted | 350.6 | 355.453041 | 4.853041 | 4.853041 | 1.38421 | 1.38421 | 2 | 0.295591 | 351.2 | 358.4 | False | 53.5 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_plus_non_target_bulk | equal_weight | 350.6 | 361.594126 | 10.994126 | 10.994126 | 3.135803 | 3.135803 | 2 | 0.302348 | 358.4 | 366.0 | False | 80.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_plus_non_target_bulk | correlation_weighted | 350.6 | 355.38086 | 4.78086 | 4.78086 | 1.363622 | 1.363622 | 2 | 0.279582 | 351.2 | 358.4 | False | 59.5 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_only | equal_weight | 378.2 | 387.491683 | 9.291683 | 9.291683 | 2.456817 | 2.456817 | 2 | 0.308593 | 378.0 | 400.2 | True | 74.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_only | correlation_weighted | 378.2 | 379.32147 | 1.12147 | 1.12147 | 0.296528 | 0.296528 | 2 | 0.294253 | 378.0 | 381.2 | True | 53.5 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_plus_non_target_bulk | equal_weight | 378.2 | 387.352316 | 9.152316 | 9.152316 | 2.419967 | 2.419967 | 2 | 0.302753 | 378.0 | 400.2 | True | 80.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_plus_non_target_bulk | correlation_weighted | 378.2 | 379.355559 | 1.155559 | 1.155559 | 0.305542 | 0.305542 | 2 | 0.279407 | 378.0 | 381.2 | True | 59.5 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_only | equal_weight | 399.2 | 409.452727 | 10.252727 | 10.252727 | 2.568318 | 2.568318 | 2 | 0.308593 | 395.6 | 428.0 | True | 74.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_only | correlation_weighted | 399.2 | 398.601826 | -0.598174 | 0.598174 | -0.149843 | 0.149843 | 2 | 0.292267 | 395.6 | 402.8 | True | 53.5 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_plus_non_target_bulk | equal_weight | 399.2 | 409.275016 | 10.075016 | 10.075016 | 2.523802 | 2.523802 | 2 | 0.302672 | 395.6 | 428.0 | True | 80.0 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_plus_non_target_bulk | correlation_weighted | 399.2 | 398.672488 | -0.527512 | 0.527512 | -0.132142 | 0.132142 | 2 | 0.277295 | 395.6 | 402.8 | True | 59.5 | petroleum_fraction_db_009 |
