# Book Genre Classification

*For further details, please see the README in each folder.*

## Normal_Part folder

Including codes of the crawler, the traditional ML models, and the final front end. 

### Brief summary

Accuracy of Maximum Entropy by using Excerpt and/or Summary: 

- Fiction/Nonfiction: 0.90025
- Multi-tag classification:

| Error Tolerance (among 24 tags)        | Accuracy      |
| -------------------------------------- |:-------------:|
| 0                                      | 0.223918      |
| 1                                      | 0.638676      |
| 2                                      | 0.905852      |
| 3                                      | 0.974554      |
| 4                                      |  0.994910     |

## Deep_part folder
Deep learning of CNN and LSTM. The performance was not better than MaxEnt. Details are in Jupyter notebooks in the `Deep_part` folder.

