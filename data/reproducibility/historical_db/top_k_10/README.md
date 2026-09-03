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
| BP_20 | analysis_only | correlation_weighted | 14 | 37.923729 | 50.680098 | 2.502753 | 11.152144 | -4.771758 | 19.044424 | 0.571429 |
| BP_20 | analysis_only | equal_weight | 14 | 46.89241 | 65.474938 | 3.55435 | 14.067727 | -5.399986 | 19.227472 | 0.571429 |
| BP_20 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 36.133548 | 47.420143 | 2.826315 | 10.702937 | -2.821818 | 19.192373 | 0.571429 |
| BP_20 | analysis_plus_non_target_bulk | equal_weight | 14 | 41.003259 | 55.1525 | 3.331026 | 12.271519 | -3.266129 | 20.01569 | 0.571429 |
| BP_35 | analysis_only | correlation_weighted | 14 | 31.042594 | 45.116788 | 0.950519 | 8.369931 | -6.803789 | 13.491981 | 0.714286 |
| BP_35 | analysis_only | equal_weight | 14 | 39.967669 | 60.362985 | 1.997736 | 10.999031 | -6.74336 | 12.537322 | 0.714286 |
| BP_35 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 28.163222 | 41.457576 | 1.211319 | 7.63035 | -4.9762 | 13.859776 | 0.642857 |
| BP_35 | analysis_plus_non_target_bulk | equal_weight | 14 | 33.866043 | 49.965332 | 1.806694 | 9.277494 | -4.815364 | 15.559889 | 0.642857 |
| BP_50 | analysis_only | correlation_weighted | 14 | 26.387176 | 41.105668 | -0.086796 | 6.747131 | -8.860636 | 12.149494 | 0.785714 |
| BP_50 | analysis_only | equal_weight | 14 | 35.547722 | 57.103714 | 1.037246 | 9.31014 | -8.258233 | 12.19003 | 0.785714 |
| BP_50 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 24.31793 | 37.902608 | 0.147579 | 6.266402 | -7.177084 | 11.024688 | 0.785714 |
| BP_50 | analysis_plus_non_target_bulk | equal_weight | 14 | 29.314315 | 46.915464 | 0.85287 | 7.637626 | -6.522854 | 9.308639 | 0.785714 |
| BP_65 | analysis_only | correlation_weighted | 14 | 30.2886 | 41.742739 | -1.115397 | 7.475747 | -12.663173 | 22.540385 | 0.785714 |
| BP_65 | analysis_only | equal_weight | 14 | 36.751848 | 57.52647 | 0.199849 | 9.215811 | -11.215339 | 15.348357 | 0.785714 |
| BP_65 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 27.661191 | 39.241051 | -0.724703 | 6.944858 | -10.488582 | 18.26822 | 0.785714 |
| BP_65 | analysis_plus_non_target_bulk | equal_weight | 14 | 30.803301 | 47.643145 | 0.010866 | 7.729958 | -9.597442 | 16.66632 | 0.785714 |
| BP_80 | analysis_only | correlation_weighted | 14 | 35.468013 | 49.022154 | -1.924615 | 8.103428 | -17.353389 | 20.011596 | 0.714286 |
| BP_80 | analysis_only | equal_weight | 14 | 43.307995 | 63.45874 | -0.817625 | 10.066084 | -16.667163 | 25.680275 | 0.714286 |
| BP_80 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 35.181187 | 48.031565 | -1.887831 | 8.084073 | -16.88845 | 20.001953 | 0.714286 |
| BP_80 | analysis_plus_non_target_bulk | equal_weight | 14 | 37.033058 | 53.816038 | -1.080185 | 8.535367 | -15.262044 | 19.551688 | 0.714286 |
| density_20c | analysis_only | correlation_weighted | 14 | 0.026825 | 0.041426 | -1.371081 | 2.853512 | -0.014484 | 0.00994 | 0.857143 |
| density_20c | analysis_only | equal_weight | 14 | 0.026543 | 0.041219 | -1.044982 | 2.8285 | -0.011612 | 0.010836 | 0.857143 |
| density_20c | analysis_plus_non_target_bulk | correlation_weighted | 14 | 0.023078 | 0.035105 | -0.892875 | 2.474155 | -0.009853 | 0.008972 | 0.857143 |
| density_20c | analysis_plus_non_target_bulk | equal_weight | 14 | 0.026489 | 0.039162 | -0.750398 | 2.843023 | -0.008864 | 0.011976 | 0.857143 |
| saturates | analysis_only | correlation_weighted | 14 | 9.782763 | 14.131429 | 19.16056 | 23.81199 | 5.604873 | 5.794095 | 0.928571 |
| saturates | analysis_only | equal_weight | 14 | 9.291123 | 13.603403 | 17.71501 | 22.921019 | 4.626798 | 7.518845 | 0.928571 |
| saturates | analysis_plus_non_target_bulk | correlation_weighted | 14 | 8.335878 | 11.702872 | 15.359137 | 19.933698 | 4.23331 | 5.741981 | 0.928571 |
| saturates | analysis_plus_non_target_bulk | equal_weight | 14 | 9.111692 | 12.782386 | 15.995908 | 21.840006 | 3.93333 | 7.454559 | 0.928571 |

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
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_only | equal_weight | 0.8713 | 0.871024 | -0.000276 | 0.000276 | -0.031731 | 0.031731 | 10 | 0.308593 | 0.8348 | 0.9149 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_only | correlation_weighted | 0.8713 | 0.867298 | -0.004002 | 0.004002 | -0.459278 | 0.459278 | 10 | 0.345293 | 0.8348 | 0.9004 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_plus_non_target_bulk | equal_weight | 0.8713 | 0.866125 | -0.005175 | 0.005175 | -0.593935 | 0.593935 | 10 | 0.29803 | 0.8348 | 0.9004 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_plus_non_target_bulk | correlation_weighted | 0.8713 | 0.866485 | -0.004815 | 0.004815 | -0.552631 | 0.552631 | 10 | 0.328162 | 0.8348 | 0.9004 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_only | equal_weight | 91.31 | 83.886408 | -7.423592 | 7.423592 | -8.130097 | 8.130097 | 10 | 0.308593 | 57.835489 | 96.271711 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_only | correlation_weighted | 91.31 | 85.270028 | -6.039972 | 6.039972 | -6.614798 | 6.614798 | 10 | 0.354559 | 71.554633 | 96.271711 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_plus_non_target_bulk | equal_weight | 91.31 | 83.989923 | -7.320077 | 7.320077 | -8.016731 | 8.016731 | 10 | 0.302101 | 57.835489 | 96.271711 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_plus_non_target_bulk | correlation_weighted | 91.31 | 85.976612 | -5.333388 | 5.333388 | -5.840968 | 5.840968 | 10 | 0.347395 | 71.554633 | 96.271711 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_only | equal_weight | 350.6 | 365.863636 | 15.263636 | 15.263636 | 4.353576 | 4.353576 | 10 | 0.308593 | 336.4 | 399.6 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_only | correlation_weighted | 350.6 | 367.702559 | 17.102559 | 17.102559 | 4.878083 | 4.878083 | 10 | 0.295591 | 336.4 | 399.6 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_plus_non_target_bulk | equal_weight | 350.6 | 366.661547 | 16.061547 | 16.061547 | 4.58116 | 4.58116 | 10 | 0.302348 | 336.4 | 399.6 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_plus_non_target_bulk | correlation_weighted | 350.6 | 363.095265 | 12.495265 | 12.495265 | 3.563966 | 3.563966 | 10 | 0.279582 | 336.4 | 399.6 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_only | equal_weight | 378.2 | 391.866141 | 13.666141 | 13.666141 | 3.613469 | 3.613469 | 10 | 0.308593 | 352.0 | 415.8 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_only | correlation_weighted | 378.2 | 391.71599 | 13.51599 | 13.51599 | 3.573768 | 3.573768 | 10 | 0.294253 | 352.0 | 415.8 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_plus_non_target_bulk | equal_weight | 378.2 | 391.684087 | 13.484087 | 13.484087 | 3.565332 | 3.565332 | 10 | 0.302753 | 352.0 | 415.8 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_plus_non_target_bulk | correlation_weighted | 378.2 | 388.135175 | 9.935175 | 9.935175 | 2.626963 | 2.626963 | 10 | 0.279407 | 352.0 | 415.8 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_only | equal_weight | 399.2 | 413.112535 | 13.912535 | 13.912535 | 3.485104 | 3.485104 | 10 | 0.308593 | 365.8 | 438.0 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_only | correlation_weighted | 399.2 | 411.33229 | 12.13229 | 12.13229 | 3.039151 | 3.039151 | 10 | 0.292267 | 365.8 | 438.0 | True | 69.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_plus_non_target_bulk | equal_weight | 399.2 | 412.196912 | 12.996912 | 12.996912 | 3.255739 | 3.255739 | 10 | 0.302672 | 365.8 | 438.0 | True | 75.9 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_plus_non_target_bulk | correlation_weighted | 399.2 | 410.729348 | 11.529348 | 11.529348 | 2.888113 | 2.888113 | 10 | 0.277295 | 365.8 | 438.0 | True | 75.9 | petroleum_fraction_db_009 |
