ML And DL Based Brain Stroke Prediction By Analysing Structured Data
 
Abstract -   Brain Stroke continues to be a major global health issue, as it is a contributing factor to high rates of morbidity and mortality. Early detection and prevention of stroke are made possible by the potential intersection of deep learning (DL) and machine learning (ML) in the healthcare industry. The performance constraints of current methods for early stroke detection highlight the need for creative fixes. Here, we introduce a brand-new web-based tool that evaluates a person's risk of stroke based on important factors like gender and BMI. Our interface makes use of cutting-edge predictive modelling techniques to offer a user-friendly platform that overcomes the shortcomings of current approaches. After extensive testing, our suggested combination of Logistic Regression (Grid searched), SMOTE and Standard Scaler obtains a 36.7% F1 score for the minority class (stroke) and that of 96.4% for the majority class (no stroke) in our unbalanced dataset. By providing a workable solution for stroke risk assessment and bridging the gap between intricate predictive models and public accessibility, this study advances the field of health informatics. With potential applications for both individuals looking to assess their stroke risk and healthcare professionals involved in preventive care initiatives, the proposed interface holds promise for improving public health outcomes by enabling early detection and prevention of stroke. 
Index Terms - Deep Learning, Early Stroke Detection, Machine Learning, Predictive Modelling, Public Health, Stroke Risk Assessment, Web-based Interface 
I.	Introduction
Stroke is a leading cause of death and disability worldwide. According to a statement by Dr PK Singh, WHO regional director for South-East Asia, 11 million strokes occur in low-and middle-income countries. This causes 4 million deaths annually and leaves approximately 30% of survivors seriously disabled. For the 70% of survivors who recover, the likelihood of suffering further strokes is greatly increased [1]. Early detection and intervention can significantly reduce the severity of outcomes and improve patient prognosis. Stroke (also called cerebrovascular accident), is the damage to the brain because of interruption of its blood supply. A stroke occurs when a blood vessel in the brain becomes blocked or narrowed, or when a blood vessel bursts and spills blood into the brain [2]. The two main types of brain strokes are Ischemic Stroke and Hemorrhagic Stroke [3]. Since there is no effective solution for stroke, early detection can prevent various deaths [4]. If the stroke is detected or diagnosed early, the loss of death and severe brain damage can be prevented in 85% of cases [5]. The stroke starts with a ministroke which is known as a transient ischemic attack (TIA). It is the condition which reveals that the person will face a stroke within a couple of days from the occurrence of the ministroke [6]. With the advent of machine learning and deep learning, predictive models have been developed to assess the risk of stroke based on various health parameters such as BMI, Smoking Status etc. However, these models are often complex and require technical expertise to use, limiting their accessibility to the general public.
In response to this, we have developed a web-based interface that leverages machine learning to assess an individual’s vulnerability to brain stroke. This interface is designed to serve as a pre-check or primary measure for people to determine their potential risk. It is based on a model trained on a dataset from Kaggle, which was selected as the best performing model after training and testing 11 machine learning models (including grid-searched models of the traditional ML models) and 5 deep learning models across 3 types of scalers and 3 types of oversamplers each.
The aim of this paper is to present the development and evaluation of this web-based interface, and to provide a comparative analysis of the different predictive models used. We also discuss the limitations of our interface, particularly the fact that the output does not guarantee medical accuracy due to the unverified nature of the dataset. Despite this, we believe that our interface can serve as a valuable resource for individuals to assess their stroke risk and seek further medical consultation if necessary. We hope that our work will contribute to the field of health informatics and inspire further research in this area.
II.	Related works
Shoily discussed the detection of stroke disease using machine learning algorithms, which is a challenging task given the complexity of the disease and the variety of risk factors involved [7]. Chang et al. explored the application of machine learning for facial stroke detection, focusing on expressional asymmetry and mouth askew as key facial features of stroke [8]. Kashi et al. proposed a machine-learning model for automatic detection of movement compensations in stroke patients, which can be adapted for use in in-clinic and at-home exercise programs [9]. Al-Mekhlafi et al. discussed the use of deep learning and machine learning for early detection of stroke and haemorrhage, with a focus on the use of the AlexNet model and a hybrid technique between deep learning and machine learning on the Magnetic Resonance Imaging (MRI) dataset for cerebral haemorrhage [10]. Malini et al. presented a cloud-based stroke prediction system to identify approaching stroke using a machine learning approach, with the major goal to accurately detect a stroke and alert the doctor or carer as soon as possible [11]. Tusher et al. developed a system through which brain stroke can be predicted earlier and very firstly, using some classification algorithms such as Logistic Regression, Classification and Regression Tree, K-Nearest Neighbor and Support Vector Machine [12]. Raja et al. discussed the prognosis of stroke using machine learning algorithms, with the main objective to develop an effective stroke detection health care service to predict stroke in better performance [13]. Together, the studies covered above point out a number of shortcomings in the state of stroke detection with machine learning algorithms today. These drawbacks include the need for manual model adaptation for use in clinical and at-home settings, difficulties in achieving early detection using MRI datasets, and reliance on particular facial features for detection, which may not capture all cases of stroke. Concerns exist over the prompt notification of medical personnel or carers as well as the general efficacy of predictive models in correctly identifying stroke cases.
Most common method of detecting stroke are reading CT scans and MRI scans [14-20]. The main reason for which the models can read brain scans is because of the colour differences, that is different tissues of the brain are shown with a different shade [15]. From diagnosis to prevention and treatment of stroke, ML & DL algorithms have helped us a lot. Available algorithms can both detect and classify the stroke. Different models show different accuracies in different types of scans [14, 16-19]. So, we can conclude that no fixed algorithm can be expected to give high-end results for every scan. We carefully read and analysed the types of algorithms, the type of preprocessing, and the type of datasets that these papers use to find the find solution. For this various 3D image reading algorithms have helped us to find the best solution and hence the most accurate results [14, 20].
III.	Methodology
As an overview, our system consists of multiple important steps. These include using basic data preprocessing techniques like data imputation and data splitting in conjunction with the application of one-hot encoding to multiple features. The cleaned data is then fed into multiple combinations of scaling strategies, oversampling strategies, and machine learning (ML) and deep learning (DL) models. Based on the F1 Score metrics for the minority class, these strategies are evaluated. The best performing model is then chosen and used to build and implement a web-based interface with Streamlit. 
The data chosen for this study was taken from Kaggle and it had several lifestyle variables which can be vital for the early prediction of Stroke such as gender, age, BMI, smoking status etc. Relevant patient information is provided by each row in the data. There were a total of 5110 observations in the raw dataset with 12 attributes before preprocessing. 


 
Figure 1: Distribution of features (a) gender (b) hypertension (c) heart disease (d) marital status (e) work type (f) residence type (g) smoke status (h) outcome of stroke, in the raw data.
i.	Data preparation
The initial step in our data preparation process was to eliminate the “id” column from the dataset. This was due to the absence of any additional data that could be mapped to this column, and to avoid potential confusion during model training.
Subsequently, we conducted a thorough examination of the dataset to further enhance its usability. As part of this process, we removed a single entry in the “other” gender category. This decision was made to streamline the data preprocessing phase, and given that this single entry was unlikely to significantly impact the overall dataset of 5110 observations.
Following this, we transformed the categorical values in the dataset into numerical data. This was achieved using the “label_encoder” from the scikit-learn Python library. The labels which were transformed for the given attributes, are listed in Table 1.
Table 1: One-hot encoding for the listed attributes.
Attribute	Original	Changed
Gender	Male	1
	Female	0
Ever Married	Yes	1
	No	0
Residence Type	Urban	1
	Rural	0

In the data preparation process, we further enhanced the dataset by applying one-hot encoding to the “work_type” and “smoking_status” columns. This was accomplished using the “get_dummies” function from the pandas library in Python. One-hot encoding is a process of converting categorical data variables so they can be provided to machine learning algorithms to improve predictions. With one-hot, we convert each categorical value into a new categorical value and assign a binary value of 1 or 0. Each integer value is represented as a binary vector. This step was crucial in ensuring that our machine learning models could process the data effectively. All the binary vectors are orthogonal to each other. This process makes the representation of categorical data more expressive and easier for the algorithms to understand. As said earlier, the “get_dummies” method is used for this purpose. This method converts the categorical variable into dummy/indicator variables. 
ii.	Data imputation
During our detailed examination of the dataset, we noticed that the "bmi" column contained approximately 4% null values. To handle these missing values, we employed a Random Forest Regressor for imputation. This decision was based on the distribution of the null values, which were scattered throughout the dataset rather than being concentrated in a specific region. 
We have included scatter plots to visually represent this process. The first plot displays the original "bmi" values, with null values marked in red and existing values in blue. The second plot shows the dataset after imputation, with the newly imputed "bmi" values indicated in green. These plots provide a clear visual representation of the imputation process and its impact on the "bmi" column in our dataset.
 
(a)
 
(b)
Figure 2: (a) Plot of BMI values in original dataset (b) Plot of BMI values after imputation.
In addition to handling null values, we also conducted a thorough statistical analysis of the "bmi" column both before and after the imputation process. This involved calculating the mean, median, and mode of the "bmi" values. Our analysis confirmed that the imputation process did not significantly alter these statistical measures, indicating that our imputation method was effective and did not introduce any significant bias into the dataset. The specific values for the mean, median, and mode before and after the imputation process are provided in the plot further. This rigorous statistical analysis ensures the integrity of our data preparation process and the reliability of the subsequent predictive models.
 
Figure 3: Mean, median and mode values before and after the imputation.
iii.	Correlation analysis
To further understand the relationships among the attributes in our dataset, we conducted a correlation analysis. This was visualized using a heatmap, which provides a color-coded representation of the correlation coefficients between each pair of attributes. 
However, our analysis did not reveal any significant correlations between any two or three attributes in the dataset. Given this, we decided to retain all the columns in the dataset and did not drop any additional columns. The heatmap illustrating these correlations is provided in Figure 4. This comprehensive analysis ensures that our models are trained on a dataset that accurately represents the complexity and diversity of the factors contributing to stroke risk.
 
Figure 4: Correlation heatmap in different data columns

iv.	Data splitting
Following the data preparation, we segregated the attributes into features and target variables in the code. We then divided the data into training and testing sets, allocating 80% of the data for training and 20% for testing. This was accomplished using the "train_test_split" function from the scikit-learn Python library, with "random_state" set to 42 and "stratify" set to y.
The "stratify" parameter is particularly important in our context. When we set `stratify` to y, we ensure that the distribution of labels (i.e., the target variable) remains consistent in both the training and testing sets. This is crucial when dealing with imbalanced datasets such as ours in this case, where certain labels might be underrepresented. We had only 249 values of “1” in our stroke (target) attribute before splitting, among the total of 5109 observations, making it the minority class. By preserving the original distribution of labels, we can improve the generalizability of our models and prevent them from being biased towards the majority class. This approach enhances the reliability of our subsequent analyses and findings.
 
Figure 5: Data samples for each class before and after splitting
From the figure above, it can be observed that the ratio of the 1 stroke values and the 0 stroke values among the data before splitting and the data used for the training is coherent.




v.	Models and techniques
Our comparative analysis incorporated a diverse range of models, spanning both traditional Machine Learning algorithms and several Deep Learning architectures.
The choice of these particular models was driven by their widespread use and proven performance in various machine-learning tasks. The traditional Machine Learning models were selected for their interpretability and robustness, while the Deep Learning models were chosen for their ability to model complex, non-linear relationships. Furthermore, we employed grid search techniques
to try and optimize the hyperparameters of the Machine Learning models, in order to extract the maximum predictive power from these algorithms.
 
Figure 6: A figure depicting Different types of machine learning and deep learning models used in this study.
In addition to their individual strengths, the combination of these models allowed us to explore a broad spectrum of the machine learning landscape. This comprehensive approach ensured that our analysis was not limited to a specific type of model or learning algorithm, thereby enhancing the robustness and generalizability of our findings. The selection of these models is detailed in the Table 2.







 
Table 2: A table depicting different types of models and the parameters on their initialisation used in this study. 
Model Name	Parameters on initialisation
Logistic Regression (Normal)	random_state = 42,
class_weight=‘balanced’
Logistic Regression (Gridsearched)	random_state = 42, class_weight=‘balanced’
param_grid_lr = {‘penalty’: [‘l1’, ‘l2’],
‘C’: [0.001, 0.01, 0.1, 1, 10],
‘solver’: [‘liblinear’, ‘saga’],
‘max_iter’: [100, 200, 300]}
cv = 5
Decision Trees (Normal)	random_state = 42, class_weight=‘balanced’
Decision Trees (Gridsearched)	random_state = 42, class_weight=‘balanced’
param_grid_dt = {‘criterion’: [‘gini’, ‘entropy’],
‘max_depth’: [None, 10, 20, 30, 40],
‘min_samples_split’: [2, 5, 10],
‘min_samples_leaf’: [1, 2, 4]}
cv = 5
Random Forest (Normal)	random_state = 42, class_weight=‘balanced’
Random Forest (Gridsearched)	random_state = 42, class_weight=‘balanced’
param_grid_rf = {‘n_estimators’: [50, 100, 200],
‘max_depth’: [None, 10, 20],
‘min_samples_split’: [2, 5, 10],
‘min_samples_leaf’: [1, 2]}
cv = 5
SVM (Support Vector Machines) (Normal)	random_state = 42, class_weight=‘balanced’
SVM (Support Vector Machines) (Gridsearched)	random_state = 42, class_weight=‘balanced’
param_grid_svm = {‘C’: [0.1, 1, 10],
‘kernel’: [‘linear’, ‘rbf’, ‘poly’],
‘gamma’: [‘scale’, ‘auto’]}
cv = 5
Gaussian Naïve Bayes	Nil
K-Nearest Neighbours (Normal)	Nil
K-Nearest Neighbours (Gridsearched)	param_grid_knn = {‘n_neighbors’: [3, 5, 7, 9],
‘weights’: [‘uniform’, ‘distance’],
‘p’: [1, 2]}
cv = 5
Artificial Neural Network	ann_model = Sequential([
Dense(128, activation=‘relu’, input_dim=X_train_scaled.shape[1]),
Dropout(0.5),
Dense(64, activation=‘relu’),
Dense(1, activation=‘sigmoid’)
])
Compiled with: optimizer=‘adam’, loss=‘binary_crossentropy’, metrics=[‘accuracy’]
1D – Convolutional Neural Network	Reshaping the input data.
cnn_model = Sequential([
Conv1D(32, 3, activation=‘relu’, input_shape=X_train_cnn.shape[1:]),
MaxPooling1D(2),
Flatten(),
Dense(128, activation=‘relu’),
Dropout(0.5),
Dense(1, activation=‘sigmoid’)
])
Compiled with: optimizer=‘adam’, loss=‘binary_crossentropy’, metrics=[‘accuracy’]
Bi-directional Recurrent Neural Network	birnn_model = Sequential([
Bidirectional(LSTM(64, activation=‘relu’, return_sequences=True), input_shape=X_train_rnn.shape[1:]),
Dropout(0.5),
Bidirectional(LSTM(32, activation=‘relu’)),
Dropout(0.5),
Dense(1, activation=‘sigmoid’)
])
Compiled with: optimizer=‘adam’, loss=‘binary_crossentropy’, metrics=[‘accuracy’]
Long Short Term Memory (LSTM)	lstm_model = Sequential([
LSTM(64, activation=‘relu’, input_shape=X_train_rnn.shape[1:], return_sequences=True),
Dropout(0.5),
LSTM(32, activation=‘relu’),
Dropout(0.5),
Dense(1, activation=‘sigmoid’)
])
Compiled with: optimizer=‘adam’, loss=‘binary_crossentropy’, metrics=[‘accuracy’]
Gated Recurrent Unit (GRU)	gru_model = Sequential([
GRU(64, activation=‘relu’, input_shape=X_train_rnn.shape[1:], return_sequences=True),
Dropout(0.5),
GRU(32, activation=‘relu’),
Dropout(0.5),
Dense(1, activation=‘sigmoid’)
])
Compiled with: optimizer=‘adam’, loss=‘binary_crossentropy’, metrics=[‘accuracy’]
 
For all the Machine Learning models listed in Table2, we set the ‘scoring’ parameter to either ‘accuracy’ or ‘precision’ wherever applicable. The Deep Learning models, on the other hand, were trained for 25 epochs each with a batch size of 32.
Each model was trained separately on three different datasets, each corresponding to a different oversampling technique: SMOTE, ADASYN, and Random Oversampling. Following oversampling, each dataset was scaled using three different scalers: MinMax, Robust, and Standard Scaler. This process was undertaken to identify the optimal combination of scaler, oversampler, and model to achieve the best performance on the test dataset. This comprehensive comparative study truly explored the breadth and depth of possibilities in our modelling approach.
 
 
Figure 7: Flowchart showing the combination of scalers, oversamplers and models used.
 
IV.	Experimental Results & discussion
I.	Experimental Setup & Dataset 
The raw dataset chosen for our study was the Stroke Prediction dataset from Kaggle. Using input criteria including id, gender, age, various anomalies, marital status, work type, residence type, average glucose level, BMI and smoking status, this dataset is used to predict a patient’s risk of having a stroke. Relevant patient information is provided by each row in the data. There were a total of 5110 observations in the raw dataset with 12 attributes originally. These factors were all relevant in stroke prediction [21-30]. This dataset was pre-processed and used to train various models after dividing it into training and testing data.
The constitution of the raw data is shown in Fig. 1.
II.	Results and Discussion
Our study involved a comprehensive comparative analysis of 144 cases, comprising 99 cases using 11 Machine Learning (ML) models and 45 cases using 5 Deep Learning (DL) models. Each model was trained and tested across three different oversampling techniques (SMOTE, ADASYN, and Random Oversampling) and scaled using three different scalers (MinMax, Robust, and Standard Scaler). 
The results of our analysis revealed significant variations in the performance of the different models, oversampling techniques, and scalers. 
 
Table 3: Comparison of accuracy for machine learning models
 
Table 4: Comparison of accuracy for deep learning models
 
Table 5:  Comparison of Accuracy for all the best performing models from our paper
 

 
Each combination of model, oversampler, and scaler produced unique results, underscoring the complexity of stroke prediction and the importance of selecting the appropriate techniques for each specific task.
The diversity of results obtained in this study highlights the importance of conducting a comprehensive comparative analysis when developing predictive models. By testing a wide range of models, oversampling techniques, and scalers, we were able to identify the combinations that performed best on our test dataset.

 
  
Figure 8: Confusion matrix for the final model, on the final Scaler and Oversampler used in our web tool 
Our dataset had an inherent class imbalance between instances of strokes (1) and non-strokes (0), so we prioritised the F1 Score for the positive class (F1_1) as a crucial metric when selecting the model to use for our web application's development. The combination that yielded the highest F1_1 (on the minority class) score of 0.36697 was the grid-searched Logistic Regression (LR_Gridsearched) model, used in conjunction with the Standard Scaler and SMOTE oversampling technique. Consequently, this combination was selected for implementation in our web application, which serves as a preliminary check for individuals seeking to assess their susceptibility to stroke.
 
Figure 9: A diagram illustrating the user-end process.
The web interface was developed using the Python library, Streamlit, which enabled us to design a user-friendly interface. We initiated the process by generating a template from the ‘streamlit-hello’ app, as per the official Streamlit documentation. This template was then customized on GitHub Codespaces to create our web interface.
Our interface provides an engaging and straightforward interface, akin to filling out an online form. Users can input their age, BMI, work type, and other relevant information via dropdown menus in the Graphical User Interface (GUI). To respect user privacy, all code is executed in real-time without any backend database, ensuring that no user data is collected or stored.
The user’s input is processed through the same scaler used during model training, and the scaled inputs are then fed into the selected model. The model’s output is presented to the user in a clear and comprehensible manner, enabling them to take appropriate action based on the results.
Our web interface is hosted and deployed via Streamlit’s Share platform, also known as the Streamlit Community Cloud. Any cookies associated with the interface are managed by the platform, and their data privacy policy applies. The authors do not store any user data in any form from the interface.
A diagram illustrating the user-end process is provided in Fig 9. The web interface can be accessed via https://strokedetection.streamlit.app/.
V.	Conclusion
In conclusion, this study has successfully developed and evaluated a web-based interface that allows individuals to assess their vulnerability to brain stroke, a leading global cause of death. The interface is grounded on a model trained on a Kaggle dataset, with rigorous testing and training conducted on 11 machine learning models and 5 deep learning models, each with different types of scalers and oversamplers. The best-performing model was ultimately selected for the interface. This research not only provides a valuable tool for early detection and prevention of brain stroke but also contributes to the field of health informatics by offering a comparative analysis of various predictive models. It underscores the potential and effectiveness of machine learning in developing practical health interfaces, paving the way for future advancements in this field. The hope is that this work will inspire further research and development in the application of machine learning for health informatics, ultimately leading to more effective and accessible health care solutions.











 
 
References
[1] Prevent brain stroke (2019) World Health Organization. Available at: https://www.who.int/southeastasia/news/detail/29-10-2016-prevent-brain-stroke 
[2] Stroke. (n.d.). National Institute of Neurological Disorders and Stroke. https://www.ninds.nih.gov/health-information/disorders/stroke#toc-who-is-more-likely-to-have-a-stroke-
[3] Louis R. Caplan, Roger P. Simon, Sara Hassani,  Chapter 27 - Cerebrovascular disease—stroke∗, Neurobiology of Brain Disorders (Second Edition), Academic Press, 2023, Pages 457-476, ISBN 9780323856546, https://doi.org/10.1016/B978-0-323-85654-6.00044-7.
[4] B. Kim, N. Schweighofer, J. P. Haldar, R. M. Leahy, and C. J. Winstein, “Corticospinal tract microstructure predicts distal arm motor improvements in chronic stroke,” Journal of Neurologic Physical Therapy, vol. 45, no. 4, pp. 273–281, 2021.
[5] M. Lee, J. Ryu, and D. Kim, “Automated epileptic seizure waveform detection method based on the feature of the mean slope of wavelet coefficient counts using a hidden Markov model and EEG signals,” ETRI Journal, vol. 42, no. 2, pp. 217–229, 2020.
[6] Kaur, M., Sakhare, S. R., Wanjale, K., & Akter, F. (2022, April 11). Early Stroke Prediction Methods for Prevention of Strokes. Behavioural Neurology; Hindawi Publishing Corporation. 
[7] Shoily, T. I., Islam, T., Jannat, S., Tanna, S. A., Alif, T. M., & Ema, R. R. (2019, July). Detection of stroke disease using machine learning algorithms. In 2019 10th International Conference on Computing, Communication and Networking Technologies (ICCCNT) (pp. 1-6). IEEE.
[8] Chang, C. Y., Cheng, M. J., & Ma, M. H. M. (2018, November). Application of machine learning for facial stroke detection. In 2018 IEEE 23rd International Conference on Digital Signal Processing (DSP) (pp. 1-5). IEEE.
[9] Kashi, S., Polak, R. F., Lerner, B., Rokach, L., & Levy-Tzedek, S. (2020). A machine-learning model for automatic detection of movement compensations in stroke patients. IEEE Transactions on Emerging Topics in Computing, 9(3), 1234-1247.
[10] Al-Mekhlafi, Z. G., Senan, E. M., Rassem, T. H., Mohammed, B. A., Makbol, N. M., Alanazi, A. A., ... & Ghaleb, F. A. (2022). Deep learning and machine learning for early detection of stroke and haemorrhage. Computers, Materials and Continua, 72(1), 775-796.
[11] Malini, T., Deepalakshmi, M., Dhivyaa, B., Karthikeswari, P., & Kavipriya, N. (2022, June). Advanced Stroke Detection and Alert System using Machine Learning. In 2022 7th International Conference on Communication and Electronics Systems (ICCES) (pp. 1084-1089). IEEE.
[12] Tusher, A. N., Sadik, M. S., & Islam, M. T. (2022, December). Early Brain Stroke Prediction Using Machine Learning. In 2022 11th International Conference on System Modeling & Advancement in Research Trends (SMART) (pp. 1280-1284). IEEE.
[13] K. S. R. S, B. Chandra, K. Kausalya, C. RM and G. R. V, "Prognosis of Stroke using Machine Learning Algorithms," 2023 7th International Conference on Computing Methodologies and Communication (ICCMC), Erode, India, 2023, pp. 1-6, doi: 10.1109/ICCMC56507.2023.10084158.
[14] Singh, S. P., Wang, L., Gupta, S., Goli, H., Padmanabhan, P., & Gulyás, B. (2019). 3D Deep Learning on Medical Images: A Review. Sensors, 20(18), 5097
[15] M. Chawla, S. Sharma, J. Sivaswamy and L. T. Kishore, "A method for automatic detection and classification of stroke from brain CT images," 2009 Annual International Conference of the IEEE Engineering in Medicine and Biology Society, Minneapolis, MN, USA, 2009, pp. 3581-3584, doi: 10.1109/IEMBS.2009.5335289.
[16] Manisha Sanjay Sirsat, Eduardo Fermé, Joana Câmara  “Machine Learning for Brain Stroke: A Review”, Journal of Stroke and Cerebrovascular Diseases, Volume 29, Issue 10, 2020, 105162, ISSN 1052-3057, https://doi.org/10.1016/j.jstrokecerebrovasdis.2020.105162.
[17] A.M. Barrett, Olga Boukrina, Soha Saleh, “Ventral attention and motor network connectivity is relevant to functional impairment in spatial neglect after right brain stroke”, Brain and Cognition, Volume 129, 2019, Pages 16-24, ISSN 0278-2626, https://doi.org/10.1016/j.bandc.2018.11.013.
[18] David A. Brenner, Rich M. Zweifler, Camilo R. Gomez, Brett M. Kissela, Deborah Levine, George Howard, Bruce Coull, Virginia J. Howard, “Awareness, Treatment, and Control of Vascular Risk Factors among Stroke Survivors”, Journal of Stroke and Cerebrovascular Diseases, Volume 19, Issue 4, 2010, Pages 311-320, ISSN 1052-3057, https://doi.org/10.1016/j.jstrokecerebrovasdis.2009.07.001.
[19] Stephen Bacchi, Toby Zerner, Luke Oakden-Rayner, Timothy Kleinig, Sandy Patel, Jim Jannes, “Deep Learning in the Prediction of Ischaemic Stroke Thrombolysis Functional Outcomes: A Pilot Study”, Academic Radiology, Volume 27, Issue 2, 2020, Pages e19-e23, ISSN 1076-6332, https://doi.org/10.1016/j.acra.2019.03.015.
[20] Puttagunta, M., Ravi, S. Medical image analysis based on deep learning approach. Multimed Interfaces Appl 80, 24365–24398 (2021), https://doi.org/10.1007/s11042-021-10707-4.
[21] Haley, Michael J., and Catherine B. Lawrence. "Obesity and stroke: Can we translate from rodents to patients?." Journal of Cerebral Blood Flow & Metabolism 36.12 (2016): 2007-2021.
[22] Miller, J., Kinni, H., Lewandowski, C., Nowak, R., & Levy, P. (2014). Management of hypertension in stroke. Annals of emergency medicine, 64(3), 248-255.
[23] Cipolla, Marilyn J., David S. Liebeskind, and Siu-Lung Chan. "The importance of comorbidities in ischemic stroke: Impact of hypertension on the cerebral circulation." Journal of Cerebral Blood Flow & Metabolism 38.12 (2018): 2129-2149.
[24] Faraco, Giuseppe, and Costantino Iadecola. "Hypertension: a harbinger of stroke and dementia." Hypertension 62.5 (2013): 810-817.
[25] Shuai Zhang, Wei Zuo, Xiao-Feng Guo, Wen-Bin He, Nai-Hong Chen, Cerebral glucose transporter: The possible therapeutic target for ischemic stroke, Neurochemistry International, Volume 70, 2014, Pages 22-29, ISSN 0197-0186, https://doi.org/10.1016/j.neuint.2014.03.007.
[26] H Jørgensen, H Nakayama, H O Raaschou, T S Olsen. “Stroke in patients with diabetes. The Copenhagen Stroke Study.” 1994, Stroke,1977-1984   doi:10.1161/01.STR.25.10.1977
[27] Sifat, A.E.; Nozohouri, S.; Archie, S.R.; Chowdhury, E.A.; Abbruscato, T.J. Brain Energy Metabolism in Ischemic Stroke: Effects of Smoking and Diabetes. Int. J. Mol. Sci. 2022, 23, 8512. https://doi.org/10.3390/ijms23158512
[28] Hawkins, Brian T., Rachel C. Brown, and Thomas P. Davis. "Smoking and ischemic stroke: a role for nicotine?." Trends in pharmacological sciences 23.2 (2002): 78-82.
[29] Jinghua Wang, Xianjia Ning, Li Yang, Jun Tu, Hongfei Gu, Changqing Zhan, Wenjuan Zhang, Ta-Chen Su. “Sex Differences in Trends of Incidence and Mortality of First-Ever Stroke in Rural Tianjin, China, From 1992 to 2012”,  2014, Stroke, Vol. 45, 6   doi:10.1161/STROKEAHA.113.003899
[30] Northern Manhattan Stroke Study Collaborators, Ralph L. Sacco, Bernadette Boden-Albala, Robert Gan, Xun Chen, Douglas E. Kargman, Steven Shea, Myunghee C. Paik, W. Allen Hauser, Stroke Incidence among White, Black, and Hispanic Residents of an Urban Community: The Northern Manhattan Stroke Study, American Journal of Epidemiology, Volume 147, Issue 3, 1 February 1998, Pages 259–268
 
