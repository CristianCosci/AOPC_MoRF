# **Report risultati $AOPC_{MoRF}$**

## Tabella riepilogativa risultati

| - |Immagine completa | 100-esimo percentile | 90-esimo percentile | 60-esimo percentile | 30-esimo percentile |
|----------- |----------- |-----------: | ----------- | ----------- | ----------- |
| **Immagini non attaccate** | 0.5632 | 0.4582 | 0.3307 | 0.1938 | 0.1041 |
| **Attacco con doppia ssim (5 filtri)** | 0.4663 <br> *-17.19%* | 0.3833 <br> *-16.34%* | 0.2937 <br> *-11.18%* | 0.1880 <br> *-2.94%* | 0.1013 <br> *-2.73%*|
| **Attacco con doppia ssim (6 filtri)** | 0.4669 <br> *-17.09%* | 0.3954 <br> *-13.70%* | 0.3148 <br> *-4.81%* | 0.2209 <br> *+14.00%* | 0.1320 <br> *+26.77%*|
| **Attacco con target_ssim + ssim (5 filtri)** | 0.5311 <br> *-5.69%* | 0.4518 <br> *-1.40%* | 0.3491 <br> *+5.56%* | 0.2278 <br> *+17.54%* | 0.1312 <br> *+25.98%*|
| **Attacco con center distance + ssim** | 0.5066 <br> *-10.05%* | 0.4237 <br> *-7.53%* | 0.3201 <br> *-3.21%* | 0.2049 <br> *+5.73%* | 0.1224 <br> *+17.53%* |
| **Immagini non attaccate ablationcam** | 0.5957 | 0.5950 | 0.5209 | 0.3605 | 0.1914 |
| **Attacco ablationcam doppia ssim** | 0.4755 <br> *-20.18%* | 0.4750 <br> *-20.16%* | 0.4317 <br> *-17.12%* | 0.3344 <br> *-7.24%* | 0.2002 <br> *+4.58%* |

<hr>


### Risultati senza il calcolo del percentile (full immagine)
- AOPC mean original img:  0.5632567150117938
- AOPC mean ssim_results_8_28 img:  0.4663839017031545
    - AOPC drawdown for ssim_results_8_28 : 17.198696567090348%
- AOPC mean ssim_results_6f_8_28 img:  0.4669915485613351
    - AOPC drawdown for ssim_results_6f_8_28 : 17.09081558813604%
- AOPC mean center_distance_results_8_28 img:  0.5066007359088551
    - AOPC drawdown for center_distance_results_8_28 : 10.058642461413413%


### Risultati con il 100esimo percentile
- AOPC mean original img:  0.45825254050022124
- AOPC mean ssim_results_8_28_pct100 img:  0.38332990623155117
    - AOPC drawdown for ssim_results_8_28_pct100 : 16.34963860470598%
- AOPC mean ssim_results_6f_8_28_pct100 img:  0.3954666746099521
    - AOPC drawdown for ssim_results_6f_8_28_pct100 : 13.701149549925695%
- AOPC mean center_distance_results_8_28_pct100 img:  0.4237369108883394
    - AOPC drawdown for center_distance_results_8_28_pct100 : 7.532010531617588%


### Risultati con il 90esimo percentile
- AOPC mean original img:  0.3307734695687888
- AOPC mean ssim_results_8_28_pct90 img:  0.293779251267551
    - AOPC drawdown for ssim_results_8_28_pct90 : 11.184155231513916%
- AOPC mean ssim_results_6f_8_28_pct90 img:  0.3148461614172704
    - AOPC drawdown for ssim_results_6f_8_28_pct90 : 4.815170990672843%
- AOPC mean center_distance_results_8_28_pct90 img:  0.3201371541582006
    - AOPC drawdown for center_distance_results_8_28_pct90 : 3.215589032716004%


### Risultati con il 60esimo percentile
- AOPC mean original img:  0.19380234351429293
- AOPC mean ssim_results_8_28_pct60 img:  0.18809480902864453
    - AOPC drawdown for ssim_results_8_28_pct60 : 2.945028621507596%
- AOPC mean ssim_results_6f_8_28_pct60 img:  0.22094045934401751
    - AOPC drawdown for ssim_results_6f_8_28_pct60 : -14.002986412660768%
- AOPC mean center_distance_results_8_28_pct60 img:  0.2049200967575992
    - AOPC drawdown for center_distance_results_8_28_pct60 : -5.736645409804519%


### Risultati con il 30esimo percentile
- AOPC mean original img:  0.10418364917272427
- AOPC mean ssim_results_8_28_pct30 img:  0.10133055513511215
    - AOPC drawdown for ssim_results_8_28_pct30 : 2.7385238089347657%
- AOPC mean ssim_results_6f_8_28_pct30 img:  0.13208011738344727
    - AOPC drawdown for ssim_results_6f_8_28_pct30 : -26.776244096109487%
- AOPC mean center_distance_results_8_28_pct30 img:  0.12244721010269029
    - AOPC drawdown for center_distance_results_8_28_pct30 : -17.530160514618927%

---
---
---

ATTACCO con 5 filtri su EigenCAM (utilizzando doppia ssim)

Prime 30 img:
- Mean value for ssim_not_inv: 0.6378287387954004
- Mean value for ssim: 0.15181819323835702

Su tutto il dataset (200 img):
- Mean value for ssim_not_inv: 0.6007907153107226
- Mean value for ssim: 0.15232910931110383

ATTACCO EigenCAM doppia ssim 6 filtri
Mean value for ssim_not_inv: 0.5766398371942342
Mean value for ssim: 0.18512539938092232
----------------------------------------------------------

ATTACCO con 5 filtri su GradCAM (utilizzando doppia ssim)

Mean value for ssim_not_inv: 0.6908552513429257
Mean value for ssim: 0.14617449867314306
----------------------------------------------------------

ATTACCO con 3 filtri su EigenCAM (utilizzando doppia ssim)

Mean value for ssim_not_inv: 0.7143981985049478
Mean value for ssim: 0.09607736269632976
----------------------------------------------------------

ATTACCO con 3 filtri su EigenCAM (utilizzando doppia center_distance + ssim)

Mean value for ssim_not_inv: 0.8602444827304809
Mean value for ssim: 0.07587910294532776


ATTACCO con 5 filtri su EigenCAM (utilizzando doppia center_distance + ssim)
Mean value for ssim_not_inv: 0.829136974704341
Mean value for ssim: 0.10992976605892181