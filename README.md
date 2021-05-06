# UnsupervisedBootAL
Unsupervised Bootstrapping of Active Learning for Entity Resolution

Code-Dataset-Results for Paper

Primpeli, Anna, Christian Bizer, and Margret Keuper. "Unsupervised bootstrapping of active learning for entity resolution." European Semantic Web Conference. Springer, Cham, 2020.

Code:
Use the Thresholding_Comparison notebook to run the comparison of the different thresholding methods:
Elbow Point, Static, Otsu's, Adjusted Valley

Use the AL_Comparison notebook to run our proposed method 'boot' which uses unsupervised matching to bootstrap active learning.
In the same notebook you can set on/off the evaluation of the two bsaeline methods of our paper: no_boot, no_boot_warm

Use the Plotting notebook for visualization and plotting of the results


Datasets:
In the datasets folder you can find all data sets used for experimenation:
1. abt_buy
2. amazon_google
3. author/DBPediaAuthors_DnbDataAuthors
4. author/DBPediaAuthors_VIAFDataAuthors
5. wdc_product/headphones_headphones_catalog
6. wdc_product/phones_phones_catalog

For every dataset pair we provide the initial datasets, feature vector files and files including matching labels 
for the train and test sets.

Results:
In the results folder you can find all result files for Active Learning methods comparison as presented in the evaluation of our paper. 
