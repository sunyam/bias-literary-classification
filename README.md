# bias-literary-classification
Repository for the paper titled "Measuring the Effects of Bias in Training Data for Literary Classification" published in The 4th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature, at COLING 2020.


This repository contains the code used in all of the experiments presented in the paper - including traditional machine learning techniques, deep learning techniques and transformer-based approaches to literary classification. We also implement some data augmentation techniques used in NLP.


Additionally, we also share the metadata for the 866 digitized works used in the paper across all genre, gender, and dialog experiments. Please email the authors in case you need access to the full text data.

For more details, you can read our paper <a href="https://www.aclweb.org/anthology/2020.latechclfl-1.9">here</a>. If you use our work in your research, please cite:

```
@inproceedings{bagga-piper-2020-measuring,
    title = "Measuring the Effects of Bias in Training Data for Literary Classification",
    author = "Bagga, Sunyam  and
      Piper, Andrew",
    booktitle = "Proceedings of the The 4th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature",
    month = dec,
    year = "2020",
    address = "Online",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.latechclfl-1.9",
    pages = "74--84",
    abstract = "Downstream effects of biased training data have become a major concern of the NLP community. How this may impact the automated curation and annotation of cultural heritage material is currently not well known. In this work, we create an experimental framework to measure the effects of different types of stylistic and social bias within training data for the purposes of literary classification, as one important subclass of cultural material. Because historical collections are often sparsely annotated, much like our knowledge of history is incomplete, researchers often cannot know the underlying distributions of different document types and their various sub-classes. This means that bias is likely to be an intrinsic feature of training data when it comes to cultural heritage material. Our aim in this study is to investigate which classification methods may help mitigate the effects of different types of bias within curated samples of training data. We find that machine learning techniques such as BERT or SVM are robust against reproducing the different kinds of bias within our test data, except in the most extreme cases. We hope that this work will spur further research into the potential effects of bias within training data for other cultural heritage material beyond the study of literature.",
}
```
