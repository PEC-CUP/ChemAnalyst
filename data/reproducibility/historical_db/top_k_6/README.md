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
| BP_20 | analysis_only | correlation_weighted | 14 | 31.544783 | 42.537045 | 0.859007 | 8.936721 | -7.820074 | 13.53086 | 0.5 |
| BP_20 | analysis_only | equal_weight | 14 | 39.59064 | 54.472256 | 2.556576 | 11.709298 | -5.419027 | 15.843947 | 0.5 |
| BP_20 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 30.313087 | 39.313936 | 1.532593 | 8.701031 | -4.666025 | 17.589295 | 0.428571 |
| BP_20 | analysis_plus_non_target_bulk | equal_weight | 14 | 36.460227 | 48.381089 | 3.354748 | 11.140926 | -1.295335 | 19.097329 | 0.5 |
| BP_35 | analysis_only | correlation_weighted | 14 | 24.599595 | 37.856951 | -0.211602 | 6.393702 | -8.896243 | 9.056727 | 0.714286 |
| BP_35 | analysis_only | equal_weight | 14 | 32.163341 | 48.899163 | 1.422089 | 8.819497 | -5.715143 | 13.386733 | 0.642857 |
| BP_35 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 22.27592 | 33.77814 | 0.602823 | 5.990172 | -5.030621 | 8.185959 | 0.642857 |
| BP_35 | analysis_plus_non_target_bulk | equal_weight | 14 | 28.135882 | 42.639353 | 2.194134 | 8.039861 | -1.494154 | 13.2681 | 0.642857 |
| BP_50 | analysis_only | correlation_weighted | 14 | 23.629557 | 34.36017 | -0.4697 | 6.169607 | -8.471854 | 13.594677 | 0.714286 |
| BP_50 | analysis_only | equal_weight | 14 | 29.212951 | 45.666284 | 0.764365 | 7.781303 | -6.248658 | 12.598091 | 0.785714 |
| BP_50 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 20.860267 | 31.562502 | 0.033114 | 5.607616 | -5.81335 | 9.912116 | 0.785714 |
| BP_50 | analysis_plus_non_target_bulk | equal_weight | 14 | 23.720267 | 39.667609 | 1.620421 | 6.660246 | -1.570636 | 10.668531 | 0.785714 |
| BP_65 | analysis_only | correlation_weighted | 14 | 27.665649 | 37.551261 | -0.964565 | 6.992296 | -10.670142 | 21.080645 | 0.785714 |
| BP_65 | analysis_only | equal_weight | 14 | 31.214882 | 46.371821 | 0.171692 | 8.044884 | -8.247067 | 17.114334 | 0.785714 |
| BP_65 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 22.882486 | 32.958732 | -0.692582 | 5.921231 | -8.539508 | 14.062511 | 0.785714 |
| BP_65 | analysis_plus_non_target_bulk | equal_weight | 14 | 25.52675 | 40.983029 | 0.923247 | 6.909686 | -3.913051 | 13.993086 | 0.785714 |
| BP_80 | analysis_only | correlation_weighted | 14 | 33.539565 | 45.838832 | -1.7182 | 7.764656 | -15.192232 | 16.803443 | 0.714286 |
| BP_80 | analysis_only | equal_weight | 14 | 37.15273 | 52.201539 | -0.636367 | 8.776807 | -12.538031 | 22.834386 | 0.714286 |
| BP_80 | analysis_plus_non_target_bulk | correlation_weighted | 14 | 31.063274 | 41.551012 | -1.421142 | 7.204879 | -12.844814 | 18.702818 | 0.714286 |
| BP_80 | analysis_plus_non_target_bulk | equal_weight | 14 | 31.494457 | 45.080119 | -0.380975 | 7.451209 | -9.461588 | 17.469213 | 0.714286 |
| density_20c | analysis_only | correlation_weighted | 14 | 0.023186 | 0.036337 | -0.865166 | 2.481171 | -0.009682 | 0.009543 | 0.857143 |
| density_20c | analysis_only | equal_weight | 14 | 0.024607 | 0.036955 | -0.695545 | 2.642915 | -0.008213 | 0.013756 | 0.857143 |
| density_20c | analysis_plus_non_target_bulk | correlation_weighted | 14 | 0.020298 | 0.03072 | -0.575975 | 2.19484 | -0.006714 | 0.008018 | 0.857143 |
| density_20c | analysis_plus_non_target_bulk | equal_weight | 14 | 0.022332 | 0.034051 | -0.589797 | 2.39681 | -0.007066 | 0.009032 | 0.857143 |
| saturates | analysis_only | correlation_weighted | 14 | 9.076503 | 13.064225 | 17.192669 | 21.839798 | 4.946088 | 5.694453 | 0.857143 |
| saturates | analysis_only | equal_weight | 14 | 8.12239 | 11.263298 | 13.548618 | 19.066987 | 3.228974 | 6.330688 | 0.857143 |
| saturates | analysis_plus_non_target_bulk | correlation_weighted | 14 | 7.635919 | 10.03752 | 11.51944 | 17.022081 | 2.757387 | 5.464949 | 0.928571 |
| saturates | analysis_plus_non_target_bulk | equal_weight | 14 | 7.285554 | 10.328128 | 12.397588 | 17.376998 | 2.899575 | 5.190498 | 0.928571 |

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
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_only | equal_weight | 0.8713 | 0.865039 | -0.006261 | 0.006261 | -0.718538 | 0.718538 | 6 | 0.308593 | 0.8348 | 0.9004 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_only | correlation_weighted | 0.8713 | 0.866571 | -0.004729 | 0.004729 | -0.542722 | 0.542722 | 6 | 0.345293 | 0.8348 | 0.9004 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_plus_non_target_bulk | equal_weight | 0.8713 | 0.864277 | -0.007023 | 0.007023 | -0.806054 | 0.806054 | 6 | 0.29803 | 0.8348 | 0.9004 | True | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | density_20c | analysis_plus_non_target_bulk | correlation_weighted | 0.8713 | 0.865272 | -0.006028 | 0.006028 | -0.691849 | 0.691849 | 6 | 0.328162 | 0.8348 | 0.9004 | True | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_only | equal_weight | 91.31 | 86.110747 | -5.199253 | 5.199253 | -5.694068 | 5.694068 | 6 | 0.308593 | 71.554633 | 96.271711 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_only | correlation_weighted | 91.31 | 85.170065 | -6.139935 | 6.139935 | -6.724275 | 6.724275 | 6 | 0.354559 | 71.554633 | 96.271711 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_plus_non_target_bulk | equal_weight | 91.31 | 86.288465 | -5.021535 | 5.021535 | -5.499436 | 5.499436 | 6 | 0.302101 | 71.554633 | 96.271711 | True | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | saturates | analysis_plus_non_target_bulk | correlation_weighted | 91.31 | 85.296264 | -6.013736 | 6.013736 | -6.586065 | 6.586065 | 6 | 0.347395 | 71.554633 | 96.271711 | True | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_only | equal_weight | 350.6 | 365.720888 | 15.120888 | 15.120888 | 4.31286 | 4.31286 | 6 | 0.308593 | 343.0 | 399.6 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_only | correlation_weighted | 350.6 | 363.965998 | 13.365998 | 13.365998 | 3.812321 | 3.812321 | 6 | 0.295591 | 351.2 | 378.4 | False | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_plus_non_target_bulk | equal_weight | 350.6 | 365.455692 | 14.855692 | 14.855692 | 4.23722 | 4.23722 | 6 | 0.302348 | 343.0 | 399.6 | True | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_20 | analysis_plus_non_target_bulk | correlation_weighted | 350.6 | 363.430661 | 12.830661 | 12.830661 | 3.65963 | 3.65963 | 6 | 0.279582 | 351.2 | 378.4 | False | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_only | equal_weight | 378.2 | 392.755682 | 14.555682 | 14.555682 | 3.848673 | 3.848673 | 6 | 0.308593 | 375.6 | 415.8 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_only | correlation_weighted | 378.2 | 390.057199 | 11.857199 | 11.857199 | 3.135166 | 3.135166 | 6 | 0.294253 | 378.0 | 411.2 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_plus_non_target_bulk | equal_weight | 378.2 | 392.386885 | 14.186885 | 14.186885 | 3.751159 | 3.751159 | 6 | 0.302753 | 375.6 | 415.8 | True | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_35 | analysis_plus_non_target_bulk | correlation_weighted | 378.2 | 389.296925 | 11.096925 | 11.096925 | 2.934142 | 2.934142 | 6 | 0.279407 | 378.0 | 411.2 | True | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_only | equal_weight | 399.2 | 414.777894 | 15.577894 | 15.577894 | 3.902278 | 3.902278 | 6 | 0.308593 | 395.6 | 438.0 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_only | correlation_weighted | 399.2 | 411.503371 | 12.303371 | 12.303371 | 3.082007 | 3.082007 | 6 | 0.292267 | 395.6 | 438.0 | True | 67.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_plus_non_target_bulk | equal_weight | 399.2 | 414.342538 | 15.142538 | 15.142538 | 3.793221 | 3.793221 | 6 | 0.302672 | 395.6 | 438.0 | True | 73.166667 | petroleum_fraction_db_009 |
| petroleum_fraction_db_007 | petroleum_fraction_db_007 | BP_50 | analysis_plus_non_target_bulk | correlation_weighted | 399.2 | 409.235379 | 10.035379 | 10.035379 | 2.513872 | 2.513872 | 6 | 0.277295 | 395.6 | 428.0 | True | 73.166667 | petroleum_fraction_db_009 |
