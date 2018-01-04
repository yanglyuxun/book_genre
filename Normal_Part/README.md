# Multi-label Book Genre Classification

Normal part with several traditional NL models


A more organized version of this can be downloaded here:

https://drive.google.com/file/d/1kGzjDV8MGKU6m9JHGCNyYBQAOHn2KPmy/view?usp=sharing

## Data:

- ./booklistall_v3.csv
"The csv table of the information of the books data we have collected, used in R and MATLAB"

- ./MAIN_Python3/txt_pro.zip
"The txt files including all preprocessed words data, used in Python and R"

- ./MAIN_Python3/booklist.pc
- ./MAIN_Python3/sum_exc_gen.pc
./MAIN_Python3/sum_exc_words.pc
"They are all pickle format (with gzip compression) data of the book list, used in Python"


## Codes:

./preprocessing.R
"R codes for preprocessing"

./Crawler_Python3/
"All the python codes of the crawler for collecting data, where nprspider.py is the first crawler, and the other 2 files are used to add more data"

./MAIN_Python3/1_from_sep_words.py
"Save the words data (from preprocessing) to a pickle format, so that Python can easily read it"

./MAIN_Python3/2_fic_classifier.py
"The codes to do the experiments of using different models to classify Fication/Nonfiction so that the accuracies can be compared"

./MAIN_Python3/2_fic_maxent.py
"Write our own codes for MaxEnt, 'class my_MaxEnt()', and trained and tested it for Fiction/Nonfiction"

./MAIN_Python3/3_tag_classifier.py
"Make the whole model for all 24 tags, as a 'class multiclassify()'."

./MAIN_Python3/4_image.py
"The SVM model for cover images, which was dropped for poor accuracy"

./MAIN_Python3/5_dash.py
"The Dash codes of the front end, which is running on the server"

./MAIN_Python3/WHOLE_MODEL.py
"The file to include all packages and functions needed in a complete prediction, used by ./MAIN_Python3/5_dash.py. It is also on the server of the front end".

./MAIN_Python3/analyze.py
"used to produce some results of analyzing"

./MAIN_Python3/sparse_class.py
"a sparse class that is used to save the data matrix, used in the above python files"

./Read_Name_Txt/Read_Name_Txt.m
"The MATLAB codes for supervised LDA"

./Read_Name_Txt/DependencyLDA_MATLAB/
"The MATLAB package for supervised LDA, used in the above codes"
