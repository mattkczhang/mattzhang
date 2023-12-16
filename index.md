---
layout: default
---

# Education
---
- #### <img src="https://github.com/mattkczhang/mattzhang/assets/94136772/ae8aac32-2cfc-4ca5-b1b9-deb9ae8e83b9" width="30" height="30" /> M.S. in Applied Data Science @ The University of Chicago 
- #### <img src="https://github.com/mattkczhang/mattzhang/assets/94136772/9d5651fc-3749-40ba-ad4f-18452ee5bfcf" width="30" height="30" /> B.A. in Economics; Political Science @ Macalester College 

# Portfolio
---
## Recommender System

### Product recommendation based on Deep Factorization Machine Model and Deep Cross Network

This project is completed when interning at MacHarry Education Consulting. It is a end-to-end project from data engineering, business intelligence, deep learning modeling and to web app deployment demo. 

[![](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/mattkczhang/transaction_project)

**Data Engineering:** Developed an OLTP through data modeling and normalizing data in third normal form to improve DML performance and data integrity. Designed and implemented ETL, denormalized data, and loaded it into a star schema to optimize the DQL performance. Created index and conducted index tuning to speed up data fetching. Created views to simplify code complexity and improve data security.

**Business Intelligence:** Built the MySQL Server-Tableau connection and designed 11 visualizations and 2 interactive dashboards monitoring customer purchasing behaviors and product popularity, helping the marketing team to identify and predict the trends and the seasonality.

**Deep Learning Modeling of Deep Factorization Machine (DFM) and Deep Cross Network (DCN):** Conducted EDA to uncover the relationships among customers’ demographic information, products’ characteristics, and transaction status, cleaned the data by converting data types and handling outliers and missing values, and developed a Deep Factorization Machine (DFM) and Deep Cross Network (DCN) model based on implicit feedback, achieving 95.5% F1 Score, 16.9% recall@10 in the DFMN and aiming to increase the revenue by 30%.

**Web App Deployment:** Developed an automatic data flow from data loading from OLAP to data cleaning and transformation and to modeling via Python, deployed the final model on a web app demo via Streamlit where employees can log in, check the customers’ profile and recommended products by entering their ID, and rerun the model on the up-to-date data, aiming to reduce the data analysis workload for 25%.

<center>
  <img width="826" alt="image" src="https://github.com/mattkczhang/mattzhang/assets/94136772/26bfc054-c0e9-47f9-84ef-dc37d09c7af1">
</center>

---
## Natural Language Processing

### Radiology Report Text Mining

I worked on this project when I interned at Inference Analytics Inc.. It aims to develop an AI-based text-mining system that extracts the keys of radiology reports and outputs structural data for UChicago Medicine.

**Preliminary Study:** Tokenized the text data, analyzed the part-of-speech, dependency, and entity annotation of sampling sentences, identified 18 sentence patterns for keywords extraction using SpaCy and Stanza, and successfully influenced stakeholders to integrate the new patterns into the NLP module, leading to a 25% increase in accuracy.

---
## Computer Vision

### Cell Image Segmentation

[![](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/mattkczhang/cell_segmentation_abzooba)

This project is completed in my UChicago Capstone collaborating with Abzooba Inc. The goal is to build a deep learning solution and a deployment demo on edge device. 

**Deep Learning Modeling of U-Net and FCN-8:** Developed an 8 times faster image segmentation tool for cellular research by building deep learning U-Net and FCN-8 models from scratch on cloud via Pytorch and deployed the best-performing U-Net on Android with an inference time 1 second per image. We achieved cross-validated 99.8% accuracy and 99.5% IoU on the final model by increasing the number of epochs, setting up early stop, and imposing image augmentation to overcome the overfitting problem and improve model generalization. 

**Edge Device Deployment:** Collaborated with cross-functional teams to build the end-to-end project pipeline from data extraction to model deployment on Abzooba’s AI/ML operation platform Xpresso.ai, leading to a 30% increase in productivity. 

<center>
  <img width="936" alt="image" src="https://github.com/mattkczhang/mattzhang/assets/94136772/9b26cd45-2707-4bc7-9377-869409accba1">
</center>

---
## Business Intelligence and Data Engineering and Big Data

### Dota2 Game Balance Analysis

[![](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/mattkczhang/Open-Dota-Analytical-Pipeline)

This project is a in-class group project at the University of Chicago Data Engineering Platform course. The goal is to build an end-to-end pipeline from OpenDota API data to ETL and data flattening transformation and to anlytical insights in BI tools. 

### Amsterdam Airbnb Pricing Analysis

[![](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/mattkczhang/Amsterdam-Airbnb-Pricing-Analysis)

This project is a in-class group project at the University of Chicago Data Visualization Techniques course in 2022 Spring. The goal is to develop a pricing system that helps both the customer and the hosts to set up a proper market price based on the details of the houses/apartments. We intend to figure out the features that are important for pricing analysis and the proper price range and develop an UI/UX design for the pricing system. 

<center><img width="1107" alt="image" src="https://user-images.githubusercontent.com/94136772/179145154-7ff1b5e5-ff4c-43bb-adfc-9947d55801c3.png"></center>

### Flight Cancellation Prediction and Features Analysis

[![](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/mattkczhang/Flight-Cancellation-Prediction-and-Features-Analysis)

This project is a in-class group project at the University of Chicago Big Data Platform course in 2022 fall. The goal is to predict flight cancellations based on weather and airline employees using Google Cloud Platform. We extracted, transformed, aggregated, and merged four datasets in total 116 GB from GCS data lake, dropped and imputed missing values, and loaded the data for EDA and feature engineering using PySpark SQL on GCP dataproc cluster. We then developed PySpark data preprocessing and random forest and gradient boosted tree machine learning pipeline to predict flight cancelation and optimized the models using grid search, achieving a 30% increase in accuracy and F1 score.
 
---
## Machine Learning

### Red Wine Quality Prediction

[![](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/mattkczhang/Red-Wine-Quality-Prediction)
[![](https://img.shields.io/badge/Render-View_on_Render-blue?logo=Render)](https://redwinequalitypred.onrender.com)

The project is a in-class group project at the University of Chicago Data Mining course in 2022 Spring. The data collects features of red wine from Vinho Verde, Portugue. The project intends to predict the red wine quality based on its chemical properties and determine which features are the best red wine quality indicators. The best-performing model is deployed on web through Streamlit and Render. 

**Machine Learning Modeling:** Both supervised and unsupervised models including K-means, linear regression, logistic regression, decision tree classification, random forests and support vector machine are developed and evaluated based on metrics like MAE, MSE, RMSE, and cross validated prediciton accuracy. The Random Forest method gives us the best accuracy score as high as 67% (which is way beyond industrial average) and the lowest MAE, MSE, RMSE. Feature importance is also calculated from the Random Forest Model. Alcohol, sulphates, volatile acidity are the most relevant features out of 10 features we have.

**Web Deployment:** With the web deployment, users can easily check the predicted score of the red wine by changing the values of different inputs/volume of different chemicals.

<center><img src="images/Red_Wine_Quality_Prediction.png"/></center>

&nbsp;

### Term Deposit Subscription Prediction

[![](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/mattkczhang/Banking-Market-Classification)

This project is a in-class group project at the University of Chicago Python for Analytics course in 2022 Spring. The bank dataset is from Portuguese banking institution and is published on UCI Machine Learning The goal is to predict if the client will subscribe to a term deposit (variable y) based on features like age and education level.

**Machine Learning Modeling:** We built KNN, logistic regression, decision tree, and random forest for prediction and evaluate them using accuracy, K-fold Accuracy, ROC, Presion Score, and Recall Score. 

### Time Series Air Pollution Prediction

[![](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/mattkczhang/Time-Series-Air-Pollution-Prediction)

This is a in-class group project at the University of Chicago Time Series Analysis and Forecasting course in 2022 Spring. The goal is to predict the future 3 months air quality based on 4 years air pollution data like PM2.5, PM10, S02, MO2, CO using Exponential Smoothing, ARIMA, Naïve, Regression, and Regression with ARMA errors. Cross validation with MSE and AICc as the evaluation metrics to check the model performance

---
## Web App 

### Weight Monitor

This web app is developed and deployed via Streamlit. It uses Google Sheet as backend database to record the daily weight. User can enter their weights everyday and a line chart is automatically generated and updated to show the trend of weights.

<center><img width="1072" alt="image" src="https://github.com/mattkczhang/mattzhang/assets/94136772/1ddf1095-ee04-4b82-9216-e4c3a091886d"></center>

---
## Economics

### The triple effect of Covid-19 on Chinese exports: First evidence of the export supply, import demand and GVC contagion effects

[![](https://img.shields.io/badge/CEPR-Link_to_Paper-blue)](https://cepr.org/system/files/publication-files/101405-covid_economics_issue_53.pdf#page=77)

This project is from my Economics summer research collaborating with Professor Felix Friedt at Macalester College in 2020. The paper estimates the overall impact of the novel Coronavirus pandemic on Chinese exports and differentiate the hypothesized `triple pandemic effect' across its three components: 1) the domestic supply shock; 2) the international demand shock; and 3) the effects of global value chain (GVC) contagion. 

**Result 1:** We find that Chinese exports are very sensitive to the severity of the global Coronavirus outbreaks. Average export elasticity estimates with respect to new Chinese and foreign destination country infections range from -2.5 to -4.6. Against a Covid-19- free counterfactual, our estimates predict that the pandemic has reduced Chinese exports by as much as 40% to 45% during the first half of 2020, but that these losses have peaked and are expected to partially recover by the end of the year. 

**Result 2:** We find that all three shocks contribute to the pandemic-induced reduction in Chinese exports, but that GVC contagion exerts the largest and most persistent influence explaining these losses. Among the three shocks, the impact of GVC contagion explains around 75% of the total reduction in Chinese exports, while the domestic supply shock in China accounts for around 10% to 15% and the international demand shock only explains around 5% to 10%. As a result of these varying transmission channels, the pandemic effects appear to be very distinct from those explaining the Great Trade Collapse in 2008-09. 

### Demand shock along the supply chain: The bullwhip effect of covid-19 in Chinese exports

[![](https://img.shields.io/badge/Macalester_Digital_Commons-Link_to_Paper-blue)](https://digitalcommons.macalester.edu/cgi/viewcontent.cgi?article=1111&context=economics_honors_projects)

This project is from my Economics Honors Thesis supervised by Professor Felix Friedt at Macalester College in 2020. The paper investigates the bullwhip effect of Covid-19 on global supply chains from the Chinese perspective. 

**Bullwhip Effect:** The bullwhip effect refers to the amplification of demand shock along the supply chain. 

**Result 1:** My baseline estimates show that a 1% increase in foreign new cases (a proxy for foreign demand shock) reduces exports of downstream products and that of upstream industries by 2.1% and 4.5% respectively. The estimates also suggest that whether industries are concentrated or not generates ambiguous effects on exports that vary from different empirical specifications. 

**Result 2:** A heterogeneity analysis suggests that the bullwhip effect is stronger in regional supply chains among geographically proximate countries and countries that are closely connected in terms of the trade volume. 

**Result 3:** A dynamic analysis shows that the outbreak of Covid-19 in foreign countries causes a lagged import substitution towards Chinese products that reverses the initially negative demand shock. Unlike the initial adverse effect, I find that the lagged import substitution does not amplify along the supply chain, but mostly affects downstream industries.

---
<center>© 2023 Kaichong (Matt) Zhang. Powered by Jekyll and the Minimal Theme.</center>
