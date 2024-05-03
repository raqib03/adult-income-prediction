# No of appearance of features for removal in the 4 RFECV experiments.

## MinMaxScaler-Accuracy
|Name of the Feature|No. of Appearance|
|:-----------------:|:---------------:|
|race               |6                |
|native-country     |4                |
|fnlwgt             |4                |
|education          |1                |
|gender             |1                |

## MinMaxScaler-f1_score
|Name of the Feature|No. of Appearance|
|:-----------------:|:---------------:|
|race               |7                |
|native-country     |6                |
|fnlwgt             |5                |
|education          |2                |
|gender             |1                |
|relationship       |1                |

## StandardScaler-Accuracy
|Name of the Feature|No. of Appearance|
|:-----------------:|:---------------:|
|race               |6                |
|native-country     |6                |
|fnlwgt             |5                |
|education          |2                |
|gender             |1                |
|relationship       |1                |

## StandardScaler-f1_score
|Name of the Feature|No. of Appearance|
|:-----------------:|:---------------:|
|race               |7                |
|native-country     |6                |
|fnlwgt             |5                |
|education          |2                |
|relationship       |2                |
|gender             |1                |
|workclass          |1                |

## Total Number of Appearance for Removal
|Name of the Feature|No. of Appearance|
|:-----------------:|:---------------:|
|race               |26               |
|native-country     |22               |
|fnlwgt             |19               |
|education          |7                |
|relationship       |4                |
|gender             |4                |
|workclass          |1                |

# Accuracies in Different Experiments
|Exp. No.|Features Removed|No Scaling|StandardScaler|MinMaxScaler|
|:------:|:--------------:|:--------:|:------------:|:----------:|
|5-1|race, native-country, fnlwgt<br>education, relationship, gender|'LightGBM'<br>0.8526581587388248|'LightGBM'<br>0.8530676312018016|'LightGBM'<br>0.8526581587388248|
|5-2|race, native-country, fnlwgt|'LightGBM'<br>0.8613253258718351|'LightGBM'<br>0.8613935712823313|'LightGBM'<br>0.8613253258718351|
|5-3|race, native-country|`'LightGBM'`<br>`0.8753156350235447`|'LightGBM'<br>0.8744284446870948|'LightGBM'<br>0.8750426533815601|
|5-4|*no feauters removed*|'LightGBM'<br>0.8737459905821333|`'LightGBM'`<br>`0.8753156350235447`|'LightGBM'<br>0.8738824814031256|