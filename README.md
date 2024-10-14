# Sentiment Spectrum: A Comparative Analysis of Emotional Impact on Brand Perception through Star Ratings

## Introduction and Background
In the era of data-driven decision-making, the importance of text analytics has become significant,
especially in understanding customer sentiment and emotions towards brands. Text analytics, as
outlined by Khan and Vorley (2017), involves the process of transforming unstructured text into
structured data for analysis, enabling businesses to gather insights from customer reviews, social
media comments, and other textual content. This analytic approach is pivotal for companies seeking
to enhance customer experience and refine their marketing, operations or product strategies.
This report focuses on a comparative analysis of customer reviews for two well-known brands.
According to Zaltman (2003), analysing customer reviews can provide deep insights into the public's
perception of a brand, which in turn can guide strategic business decisions. The objective here is to
apply advanced text analytics techniques, including machine learning, to not only categorize emotions
found in customer reviews but also to assess the overall sentiment and satisfaction level indicated by
star ratings (Nandwani and Verma, 2021).
By conducting a detailed text analysis, the aim is to uncover underlying patterns in customer emotions
and preferences, which are crucial for improving product offerings and customer service. The scope
of this analysis is significant as it directly impacts the brands ability to respond to customer needs
effectively, thereby fostering a stronger brand loyalty and improving competitive positioning in the
market.

## Methodology
The methodology of this study follows the CRISP-DM framework, which stands as the backbone of our
analytical strategy, guiding us from understanding the data to deploying the final analysis. This
methodological framework emphasizes an iterative process tailored to adapt and refine as new
insights and challenges emerge from the data (Wirth and Hipp, 2000).

![image](https://github.com/user-attachments/assets/c3a1b936-4d61-425d-8793-37ff89e76d4c)

The initial phase of our analysis, descriptive statistics and visualizations play a critical role providing
initial insights into the data distribution across different brands, star ratings and emotional
sentiments. By deploying libraries such as Seaborn and Matplotlib for visualization, we are able to
obtain a clear overview of the data distribution across various dimensions. These visualizations not
only aid in understanding the underlying patterns within the data but also highlight discrepancies and
outliers that might affect the subsequent analysis (Wickham, 2010). For example, the use of bar plots
and count plots allows for an intuitive understanding of the frequency and distribution of ratings and
emotions, which is essential for both the pre-processing and modelling stages.

Moving to the core of our text analytics, we employ supervised and semi-supervised machine learning
algorithms to classify emotions from text reviews. The selection of the SGDClassifier, a linear classifier
which implements stochastic gradient descent, is particularly suited for large datasets and text
classification due to its efficiency in handling sparse data (Bottou, 2010). The rationale behind using a
semi-supervised approach, specifically Label Spreading, is because of its capability to utilize both
labelled and unlabelled data effectively. This method leverages the abundance of unlabelled data to
enhance the learning process, thereby improving the overall model accuracy in scenarios where
labelled data may be insufficient (Zhu & Goldberg, 2009).

This approach ensures that our analysis is robust, reproducible, and aligned with the objectives of
understanding customer sentiment comprehensively. The flow of the analytical part is as followed:

![image](https://github.com/user-attachments/assets/26ff0bb6-8b68-4b0f-93e1-bf95903901ca)
![image](https://github.com/user-attachments/assets/f9098130-e963-4078-b0fe-5c10e5044f09)

### Text Data Pre-Processing
Effective text analytics begins with meticulous data cleaning and preparation, key steps in ensuring
the accuracy and reliability of subsequent analyses.

![image](https://github.com/user-attachments/assets/013712e2-e9dc-4823-998c-1733edad84c5)

These pre-processing steps are critical as they significantly reduce noise within the data, allowing the
machine learning algorithms to focus on the most impactful features (Rahman, 2019). According to
Haddi et al. (2013), such pre-processing not only improves the efficiency of the algorithms but also
enhances the overall quality and reliability of the text analytics, ensuring that the emotional sentiment
analysis is based on the most relevant and contextual aspects of the text. This rigorous preparation is
essential for accurately categorizing emotions and extracting valuable insights from customer reviews.


### Text analytics
#### Supervised & Semi-Supervised Machine Learning

##### Model Selection
In the domain of text analytics, selecting the appropriate machine learning model is crucial for
accurate classification and prediction (Minaee et al., 2021). The criteria for selecting the best classifier
include the model’s ability to handle high-dimensional data, efficiency in terms of computational
resources, and robustness against overfitting. The Stochastic Gradient Descent (SGD) classifier, utilized
in this analysis, addresses these criteria effectively. It is particularly suited for text data due to its
scalability and speed with large datasets, as well as its flexibility in optimizing different loss functions
which can be tailored to specific requirements of text classification tasks (Occhipinti et al., 2022).
Research by Bottou (2010) supports the use of SGD for sparse data typically found in text applications,
highlighting its efficiency in converging to an optimal solution faster than many batch processing
methods.

##### Modelling Approach
The semi-supervised learning approach was chosen to learn from the labelled dataset and predict
unlabelled data, thereby enhancing the training process and potentially improving model
generalization. This study employs the Label Spreading algorithm, which is effective in assigning labels
through the dataset based on similarity measures. This algorithm assumes that points which are close
to each other are likely to share the same label, thus exploiting the geometric structure of the data
space (Zhang and Zhou, 2013). By utilizing both labelled and unlabelled data, Label Spreading helps in
overcoming the limitations posed by the small size of labelled data, which is a common challenge in
practical applications.

##### Evaluation
To evaluate the performance of the supervised and semi-supervised models, several metrics are
employed. The primary metric used is the F1 score, which balances the precision and recall of the
classifier, providing a more general measure of model performance, particularly in datasets where
class imbalance might exist (Grandini et al., 2020). Additional metrics include accuracy, confusion
matrix, and a classification report which offers a detailed view of performance across different classes.
These metrics collectively offer insights into the strengths and limitations of the models.
The SGD classifier, with its ability to efficiently handle large datasets, shows robust performance in
terms of speed and scalability. However, its dependency on the correct tuning of hyper parameters
and sensitivity to feature scaling can be seen as limitations. Meanwhile, the Label Spreading model's
strength lies in its capacity to leverage unlabelled data, significantly enriching the learning process.
However, its performance heavily relies on the assumption that similar data points share the same
label (Balcan and Sharma, 2021), which may not always hold true, especially in diverse and complex
datasets.
In conclusion, both supervised and semi-supervised learning models present distinct advantages and
challenges. The chosen evaluation metrics provide a comprehensive overview of these models
performance ensuring that the insights derived from the analysis are both reliable and actionable


### Visual Analytics
The visual analytics derived from the exploratory data analysis (EDA) on the dataset provide crucial
insights into the distribution of star ratings and emotions across two well-known brands, identified
here as "H_" and "Z_". 

![image](https://github.com/user-attachments/assets/e04828ff-887c-4d6a-abd9-a0af0bb4b172)

The descriptive statistics and corresponding visualizations focus primarily on the non-text columns of
the dataset, which include star ratings and categorized emotions.

![image](https://github.com/user-attachments/assets/4418e9fe-3954-466a-9fde-89ac0829c8e4)

From the visualizations, it is evident that Brand H_ enjoys a significantly higher number of reviews
overall, as well as a noticeably larger proportion of 5-star ratings compared to Brand Z_. 

![image](https://github.com/user-attachments/assets/9fb7d61b-758c-43f5-b697-f25d22184113)

This disparity suggests a higher level of customer satisfaction or a more positive reception for Brand
H_ among reviewers. The bar graphs comparing the star ratings between the two brands clearly depict
this trend, with Brand H_ showing a prominent spike in the highest rating category.

![image](https://github.com/user-attachments/assets/f6eda283-1395-4f18-b230-d09fc19ac71c)

![image](https://github.com/user-attachments/assets/380b2591-0cbc-40a7-a965-86548c35dd67)

Among the 11% labelled data on emotions, distribution of emotions among different band offers
another layer of understanding. While both brands exhibit a similar range of emotions, the frequencies
of these emotions differ. For instance, emotions such as joy and surprise appear more frequently in
the reviews for Brand H_, potentially correlating with the higher star ratings observed. In contrast,
Brand Z_ shows a more balanced emotional distribution, which might reflect a more varied customer
experience.

These visual comparisons underscore the importance of star ratings in assessing customer emotional
sentiment and satisfaction as derived from early researches (Pashchenko et al., 2022). They provide
actionable insights that could guide both brands in enhancing their strategies for customer
engagement and satisfaction, highlighting the utility of visual analytics in business decision-making
processes.

## Results and Discussion

### Outcomes

![image](https://github.com/user-attachments/assets/045e11f7-6a55-4ae0-85f0-2122fef84a89)

The analysis employed two primary machine learning models: the Supervised SGDClassifier and the
Label Spreading model for semi-supervised learning.
1. The Supervised SGDClassifier shows the best overall performance in terms of F1 score,
precision, recall, and accuracy when only labelled data is considered. This model provides a
more reliable classification of emotions.

2. The Semi-Supervised Label Spreading on labelled data demonstrates lower effectiveness
compared to the supervised model, suggesting that the quality of labels significantly impacts
the performance of this model.

3. When the Label Spreading model is applied to a mix of labelled and unlabelled data, there is
a significant drop in the F1 score (F1 score: 0.286) and accuracy, although the precision 
appears artificially high. This decrease could be due to the model's over-reliance on the
majority class in the presence of a large volume of unlabelled data, which often misrepresents
the true predictive performance (Nigam and Ghani, 2000).

Moreover, the classification reports highlight a disparity in model capability across different emotions,
suggesting that some emotions are more challenging to classify due to overlapping linguistic cues or
less distinct expression in the text.

![image](https://github.com/user-attachments/assets/a3052d4d-2faa-423f-879f-230ad0c1c834)
![image](https://github.com/user-attachments/assets/8ce23cd9-4e64-4fd3-93e0-f7156a7d8b39)

The Supervised SGDClassifier generally provides a more balanced performance across all
emotions with relatively moderate to good precision and recall, resulting in higher F1-scores
compared to the Label Classifier.

The Label Classifier, when applied to mixed data, shows extreme values in precision and
recall—either very high precision with low recall or vice versa. This suggests issues with model
training, possibly related to how it handles the large amount of unlabelled data.


### Challenges
One of the primary challenge encountered in this project was the semi-supervised learning's
sensitivity to the initial label quality and distribution. The vast majority of the training data being
unlabelled in the Label Spreading model introduced significant uncertainty, reflected in the poor
performance metrics, particularly the low recall scores observed across most classes. Another
challenge was the imbalance in the emotional labels available in the training data, which likely
contributed to the model skewed performance towards certain emotions. This issue is evident in the
confusion matrices, where some emotions like anger and disgust show considerably varied precision
and recall.

The analysis reveals that while supervised learning approaches like the SGDClassifier provide effective
results when trained with adequately labelled data, the application of semi-supervised techniques in
environments with substantial unlabelled data requires careful consideration of data quality and
distribution. The study highlights the potential of machine learning in extracting valuable insights from
textual data but also cautions against the optimistic generalization of semi-supervised methods
without rigorous validation and tuning. Future work could focus on enhancing the label quality,
possibly through manual annotation or crowd-sourcing to provide a more solid foundation for training
semi-supervised models. Additionally, exploring more advanced techniques in natural language
processing, such as deep learning models that can capture context better, might improve both the
accuracy and reliability of emotion classification in customer reviews (Yadav and Vishwakarma, 2020).

### Insights

1. Distribution of Reviews per Brand:
   ![image](https://github.com/user-attachments/assets/45cdce99-4f17-4d82-ad14-0f675189f356)
Brand H_ significantly leads in the total number of reviews compared to Brand Z_. This suggests that Brand H_ may have a larger customer base or higher engagement level from its customers, which could be indicative of greater market presence or more
active marketing strategies.

2. Distribution of Star Ratings:
![image](https://github.com/user-attachments/assets/dcb4a443-fe14-43be-b0fd-ea1e8cfccbf9)

There is a clear preference for higher ratings, with a significant number of reviews at
the 5-star rating level. This trend suggests customer satisfaction with the services or
products offered by the brands, but also raises questions about the distribution and
possible reasons behind the high volume of positive ratings, such as selection bias or
incentivized reviews.

3. Distribution of Emotions in Reviews:
![image](https://github.com/user-attachments/assets/4206cd2b-3243-4212-86e2-a8dd55b81339)
The most frequent emotions expressed across reviews include neutral and surprise,
followed by joy, sadness, and disgust. The presence of positive emotions like surprise
at high levels correlates well with the high number of 5-star ratings, indicating positive
customer experiences.

4. Emotion Distribution by Brand:
![image](https://github.com/user-attachments/assets/f0cc222b-b9c8-40d8-bf74-eee7e93ef260)
Both brands show a variety of emotional expressions, but Brand H_ has a notably
higher frequency of neutral and positive emotions (joy and surprise) compared to
Brand Z_. This may suggest that customers of Brand H_ have more positive
experiences or that the brand's offerings are more aligned with customer
expectations.

5. Distribution of Emotions for Star Ratings 4 and 5 by Brand:
![image](https://github.com/user-attachments/assets/9e7a51e5-d675-4554-b760-dd29eebaa778)
The majority of high star ratings (4 and 5) are associated with positive emotions like
joy, which is consistent across both brands. This reaffirms that higher ratings are
typically correlated with positive emotional feedback.

6. Distribution of Emotions for Lower Star Ratings (1, 2, 3) by Brand:
![image](https://github.com/user-attachments/assets/9678baf5-023c-4244-8d3f-f3652c4e6a14)
For lower star ratings, emotions like sadness and disgust appear more prominently,
especially for 1-star ratings, indicating dissatisfaction and potential areas for
improvement

![image](https://github.com/user-attachments/assets/bd14f2d3-2743-4c0d-a46a-e01bccd19435)

![image](https://github.com/user-attachments/assets/28b7a7dc-9af1-4907-81ce-74a466004d8b)

Interestingly, neutral emotions are also noted across lower star ratings, suggesting
some customers feel indifferent or unimpressed rather than outright displeased.

The insights from these visualizations can guide strategic decisions for the brands. For Brand H_,
maintaining the factors leading to high joy and surprise in customer reviews could be crucial, while
Brand Z_ might focus on addressing the negative aspects that lead to sadness and disgust to improve
overall customer satisfaction. Both brands could benefit from investigating the root causes of neutral
emotions to convert these experiences into more positive outcomes, potentially increasing customer
loyalty and overall satisfaction.


## Conclusion and Recommendations

The analysis revealed significant insights into customer sentiment for two brands, H_ and Z_. Brand
H_ demonstrated a higher number of reviews and a tendency toward more positive emotions and
higher star ratings compared to Brand Z_. The data highlighted that positive reviews are often
accompanied by feelings of joy and surprise, whereas negative reviews are correlated with sadness,
disgust, and anger.

### Implications

Understanding these emotional distributions and their correlation with star ratings can help both
brands tailor their customer experience strategies (Dhar and Bose, 2022). For Brand H_, maintaining
the positive aspects that lead to high customer satisfaction is crucial, while Brand Z_ needs to address
the causes of negative sentiments to improve their customer relations and product or service
offerings.


### Recommendations

1. Brand H_ should continue leveraging its strengths by enhancing the features or services that
lead to high customer joy and surprise. Implementing loyalty programs or customer
appreciation initiatives could further boost positive sentiments.
2. Brand Z_ should conduct detailed customer feedback sessions to identify the specific aspects
leading to dissatisfaction. Improvements in customer service, product quality, or user
experience based on feedback could mitigate negative emotions.
3. Both brands should consider regular monitoring of customer sentiment through automated
sentiment analysis tools to quickly identify and address emerging issues before they affect a
larger segment of their customer base (Sharef et al., 2016).
4. Both brands should investigating the root causes of neutral emotions to convert these
experiences into more positive outcomes.

By acting on these insights, both brands can enhance customer satisfaction, improve loyalty, and
ultimately drive better business outcomes.




## References

1. Balcan, M.F.F. and Sharma, D., 2021. Data driven semi-supervised learning. Advances in
Neural Information Processing Systems, 34, pp.14782-14794.
2. Bottou, L., 2010. Large-scale machine learning with stochastic gradient descent.
In Proceedings of COMPSTAT'2010: 19th International Conference on Computational
StatisticsParis France, August 22-27, 2010 Keynote, Invited and Contributed Papers (pp. 177-
186). Physica-Verlag HD.
3. Bottou, L., 2010. Large-scale machine learning with stochastic gradient descent.
In Proceedings of COMPSTAT'2010: 19th International Conference on Computational
StatisticsParis France, August 22-27, 2010 Keynote, Invited and Contributed Papers (pp. 177-
186). Physica-Verlag HD.
4. Dhar, S. and Bose, I., 2022. Walking on air or hopping mad? Understanding the impact of
emotions, sentiments and reactions on ratings in online customer reviews of mobile
apps. Decision Support Systems, 162, p.113769.
5. Goldberg, A., Zhu, X., Singh, A., Xu, Z. and Nowak, R., 2009, April. Multi-manifold semisupervised learning. In Artificial intelligence and statistics (pp. 169-176). PMLR.
6. Grandini, M., Bagli, E. and Visani, G., 2020. Metrics for multi-class classification: an
overview. arXiv preprint arXiv:2008.05756.
7. Haddi, E., Liu, X. and Shi, Y., 2013. The role of text pre-processing in sentiment
analysis. Procedia computer science, 17, pp.26-32.
8. Khan, Z. and Vorley, T., 2017. Big data text analytics: an enabler of knowledge
management. Journal of Knowledge Management, 21(1), pp.18-34.
9. Minaee, S., Kalchbrenner, N., Cambria, E., Nikzad, N., Chenaghlu, M. and Gao, J., 2021. Deep
learning--based text classification: a comprehensive review. ACM computing surveys
(CSUR), 54(3), pp.1-40.
10. Nandwani, P. and Verma, R., 2021. A review on sentiment analysis and emotion detection
from text. Social network analysis and mining, 11(1), p.81.
11. Nigam, K. and Ghani, R., 2000, November. Analyzing the effectiveness and applicability of cotraining. In Proceedings of the ninth international conference on Information and knowledge
management (pp. 86-93).
12. Occhipinti, A., Rogers, L. and Angione, C., 2022. A pipeline and comparative study of 12
machine learning models for text classification. Expert systems with applications, 201,
p.117193.
13. Pashchenko, Y., Rahman, M.F., Hossain, M.S., Uddin, M.K. and Islam, T., 2022. Emotional and
the normative aspects of customers’ reviews. Journal of Retailing and Consumer Services, 68,
p.103011.
14. Rahman, A., 2019. Statistics-based data preprocessing methods and machine learning
algorithms for big data analysis. International Journal of Artificial Intelligence, 17(2), pp.44-
65.
15. Sharef, N.M., Zin, H.M. and Nadali, S., 2016. Overview and Future Opportunities of
Sentiment Analysis Approaches for Big Data. J. Comput. Sci., 12(3), pp.153-168.
16. Wickham, H., 2010. A layered grammar of graphics. Journal of computational and graphical
statistics, 19(1), pp.3-28.
17. Wirth, R. and Hipp, J., 2000, April. CRISP-DM: Towards a standard process model for data
mining. In Proceedings of the 4th international conference on the practical applications of
knowledge discovery and data mining (Vol. 1, pp. 29-39).
18. Yadav, A. and Vishwakarma, D.K., 2020. Sentiment analysis using deep learning
architectures: a review. Artificial Intelligence Review, 53(6), pp.4335-4385.
19. Zaltman, G., 2003. How customers think: Essential insights into the mind of the market.
Harvard Business Press.
20. Zhang, M.L. and Zhou, Z.H., 2013. A review on multi-label learning algorithms. IEEE
transactions on knowledge and data engineering, 26(8), pp.1819-1837.









   













