# Financial-analysis-based-on-machine-learning-and-time-series-algorithms
 We will divide the paper into two parts and code to show the principle and application of deep forest, several machine learning algorithms such as random forest, and various time series prediction methods such as ARIMA and their optimization.

The original project will be completed in **2021** on Baidu's flying Paddle platform. Here is the original website:https://aistudio.baidu.com/projectdetail/2284369  This link contains all the Chinese papers as well as the code section.



## 1.Some deep forest concepts
Deep forest is an integrated forest model jointly proposed by Zhou Zhihua and others, which is an integration of traditional forest models in breadth and depth. Although the actual operation takes up a lot of memory and the effect is not as good as deep learning, it also provides a new integrated idea for traditional machine learning.

When the differences in the learning samples are reflected enough, the effect of the integrated learning device will be improved. Therefore, the stack of deep forest is divided into two purposes: the first, multi-granularity scanning, reflecting the difference of input data, and the second, cascading forest, improving the classification ability of input data.

### Multi-Grained Scanning
Multi-Grained Scanning refers to that a number of sliding Windows of different sizes are used to take a sliding value over the original data. In this paper, there are mostly three sliding Windows, but the use of five sliding Windows is also proposed, and the effect is better than that of three. In the process of sliding window value, sampling operation can be carried out incidentally.

### Cascade forest
Each Level of the cascade forest contains several ensemble learning classifiers (in this case, the decision tree forest), which is an ensemble within an ensemble structure.

In order to reflect the diversity, two integrated learners representing several different types are used here. Each layer of the cascade forest in the figure includes two completely random forests (black) and two random forests (blue). The main difference between these two kinds of forests lies in the candidate feature space. In a completely random forest, features are randomly selected in a complete feature space to split, while in a normal random forest, the split nodes are selected in a random feature subspace by Gini coefficient.

## 2.Python reproduction of Deep forest
Combined with the iris data set commonly used in machine learning, we reproduced the deep forest model, and the results are as follows:

![3769715b3a7c4781b3e18d9e2dbd9f5a8493283a533e4665a7e510b8bb336f70](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/4be1a49d-7897-4305-a687-443d803e8f7b)

Its accuracy is as high as 0.9 or more, so the prediction is good. See the comparison and analysis below for the specific prediction and fitting degree.

## 3.Haier Zhijia June stock price forecast based on the historical real stock price

### Model input

This chapter uses the four machine learning methods introduced above to build a prediction model of stock index rise and fall. First, the previous research results are summarized, eight technical analysis indicators are selected as input variables of the model, and the input is standardized or discretized, aiming at establishing a stock index rise and fall prediction model with strong performance.

### Input feature standardization
First, the values of the 8 technical indicators are calculated based on the formula described above. Since each technical index has different dimensions and orders of magnitude, in order to ensure the reliability of the results, the original index data is standardized in this paper, and the processed technical index value is used as the feature input of the prediction model.

![image](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/554fced0-e2a1-4d03-9799-04b2b41bf432)

### Mode input improved

Since each technical indicator has its own inherent properties, for example the RSI is often used to identify overbought and oversold, with a value between 0 and 100, if the RSI>70, it indicates that the index is overbought, and the analysis of the index may fall. In different applicable scope and environment, investors usually predict the future price rise or fall trend of the stock index through different attributes of one or more technical indicators, and then guide trading. Considering the inherent properties of these technical indicators, we refer to {10} and improve the processing method of input features, which is not standardized but discretized. The specific treatment method is as follows: the continuous technical index value is transformed into a trend discrete value, with "11" to represent the rising trend of the technical index, with "-1" to represent the falling trend of the technical index. The following details how to discretize technical indicators.

### Deep forest model

The principle of the model has been described above, and the results are predicted as follows:

![image](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/52be0296-d989-44af-8ddd-c42399fc2807)

### Random forest model
Building multiple decision trees and fusing them together to get a more accurate and stable model is a combination of bagging ideas and random selection features. Random forest constructs several decision trees. When it is necessary to predict a certain sample, the prediction results of each tree in the forest are counted, and then the final result is selected from these prediction results through voting.

Randomness is reflected in two aspects, one is to take random features, and the other is to take random samples, so that every tree in the forest has similarities and differences.

![image](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/11aa24a3-e4ff-4c1f-8df7-2eabf499a005)

### LR model result prediction

![886458d3145e4c9c9c9625cb7bab1fd51e789d51d0634d7981bf490ad7aacdc5](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/a44761fe-b279-4315-ba92-82c28caf036d)

### XG-BOST model

![6e9a1cdb2a3d4fcea55d58b2b5717885db7e8f8fd41e4f35b7185a86e54eac95](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/a375fe0d-65d3-49cd-b681-ef5460e15fe0)

### SVM model

![3bccdfac7bcd4a8cafaad76220826e9bbe5d3629ba364ca8919c3a2691ce2b14](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/060ef0da-778f-485d-8ba7-1bb7aa185758)

### Result analysis
The accuracy of each model prediction is compared below. The comparison chart is as follows:

![image](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/7dbcb92f-6a76-479c-a297-f716f4170d58)


It can be seen from the experimental results that the prediction effect of deep forest is the best under this prediction model.

## 4.Regression prediction based on machine learning
### Deep forest

![d7df6687be9b40a795a967f93d00ba1dae1d29b6ddda4c94b7ff8ff3ca345a93](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/a1d6c689-ba84-4f00-8d1b-9c637fc92788)

### Random forest

![729386146cf74a848744217c80a2d8dc374427f24f7b4c5eab6ee9598a285a0a](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/f8ce7a92-50d2-4e7f-9704-ddb053bbecc6)

### Extreme forest

![9b722d53cd1e4bd9a874e7149681bd29793d03eec6ed46fc8c44fe680610dcbd](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/ff018de7-906c-4c85-ab5b-a1e9370d6c6f)

### SVM

![e6fc0d5ef5314865ae26ab9813d19d46d4380bdc901043c08ab735064703f6df](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/a0bba457-eec1-4e18-a658-3e29d721c6b8)

### Result analysis
We decided to optimize it on the basis of conventional test methods such as MSE, and used RMSSE MASE two evaluation models commonly used by teachers to evaluate it

![image](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/1c9849a1-623e-4d53-9df7-c3733a95e5d7)

Through these two evaluation criteria, the prediction and test of Haier intellectuals are used as the standard to judge the pros and cons of each model, and the results are as follows:

![image](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/42604b32-9e8e-408c-963d-ee97cd64ae2c)

According to the above test results, for regression analysis, random forest has the best prediction effect in all models.

## 5.Based on the market situation of the same industry, Haier Zhijia June stock price forecast
The change of the stock quotation of the entire industry in which the company is located can often reflect the overall trend of the company in a certain period of time, that is, the change of the company's stock price has a potential relationship with the industry in which it is located. For the only home sector where Haier Zhijia is located, this paper selects five stocks with representative significance in the concept of smart home, and the leading stocks of related concepts for analysis. The weighted average of the changes in the rise and fall of the five leading stocks. It can approximate the overall increase and decrease of the industry.

Because the price of each stock is different, the forecast process adopts the rise and fall as the unit, so that the forecast process is not affected by the difference of the unit price of each stock.

In the weight distribution of the rise and fall, this paper adopts the intuitive and convincing index market circulation rate as the only basis, that is, market circulation rate as the only standard. This choice not only considers the actual impact of each stock on the market, but also reflects the economic situation of each stock itself. Among them, the circulation value of Gree electric equipment is 287.2 billion yuan, iFlyTek is 125.3 billion yuan, Goer shares are 142.6 billion yuan, and TCL Technology is 103.8 billion yuan. After weighted according to the market circulation value of the company, the weighted weights of the rise and fall of each stock are obtained, which are 0.24978, 0.10894, 0.12398, 0.09025 and 0.42714 respectively.

### Prediction result
![62bf17a246944c8fa74cc2737fbdcc55543f8541ee814aeea66ca21cc3c89d78](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/f5e95dba-cbab-43f4-b3c0-29dfba234802)

![5b132712c85c43998c0434992522c7900c6ba5e58d83482db9c5c9f584800839](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/981b7ef8-55e0-48ba-8145-6c538a1d6b19)

![e43e5e00a4f04d50af5a6a4d81de62b314f97aea999b4a47b5a358af2196a1c6](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/a931497c-e346-4604-b36a-4a3f1bdc4176)


Compared with the data, it is obvious that the prediction effect of random forest is better than other models.

## 6.Comparison of advantages and disadvantages of deep forest model under time series
k-fold Monte-Carlo CV, as the most popular CV, is an effective way to reduce the secondary sampling bias. Given a dataset containing n data, the MCCV process first generates k subsets, each containing m data, n=km. Each data is first assigned an index between 1 and n, which serves as a framework for building a secondary sample. Through simple random secondary sampling (SRS), a pseudorandom number generator generates a random sequence of integers, ranging from 1 to n, with each integer corresponding to an element of the index. The sequence of random integers is then divided into k equally sized parts, and the data is divided into k subsets. Finally, training and iteration are performed, keeping a different subset each time to estimate EPE, and the rest for training.

However, the SRS method brings high computational costs and can also affect the test results. Inductive bias of the model and secondary sampling bias of the SRS can result in large EPEs, especially for small, unbalanced, high-dimensional data. Because SRS appear to be random, but are not really random and are not uniformly distributed, the data in such a sequence may be clustered at some arbitrary interval. To avoid this risk, the MCCV process can be repeated p times with different random seeds (usually p=50).

The reliability of MCCV increases with the increase of k and p. When k=n, MCCV is called leaveone-method (LOO). In fact, LOO is not widely used because of its higher calculation requirements and large variance.

Hierarchical MCCV first divides the data set into non-overlapping layers according to the category label vector, and then randomly divides the data into k subsets of each layer according to scale. While this ensures the same distribution of class values across all subsets, it does not ensure the same distributions of feature values. The problem of secondary sampling deviation is still unsolved.

In order to solve the above problems, K-fold optimal differential system secondary sampling (BDSCV) is proposed, which has less sampling deviation than traditional methods. The instances of the first secondary sampling are not chosen at random like traditional OSS, but according to the best difference sequence (BDS), which is a finite or infinite uniform sequence in which the number of arbitrarily proportional elements falling into any interval is proportionally converted into the interval, and the secondary sampling interval is also determined by BDS.

The following aspects will be demonstrated in detail: introducing the theory of low difference sequence (LDS) and optimal difference sequence (BDS), explaining the secondary sampling defects of traditional K-fold cross test, and proposing BDS and BDSCV.

### Process presentation
Figure 1 shows a scatter plot of three random sequences, each containing 500 elements. These three sequences obviously contain clear clustering phenomena and gaps, so the distance between each two consecutive random elements may be large or small. 

![ed8c9108f4764b149565f977e034617fa4e5b81760ab4b2f8d2c9f6795bc4129](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/2f49ffb1-115d-4e57-9e0c-ea5b6fa19a6f)

The fourth part of the two figures is a BDS containing 500 elements, in which the interval between the two continuous elements is determined by the uniform property to avoid clumps or gaps. The difference of BDS in the calculated 50 intervals is close to zero, and the variance is far smaller than the other variance, so it can be obtained that the elements of BDS should be evenly distributed as far as possible. Successive element functions are inserted as far away from other elements as possible to avoid aggregations and gaps, and sequentially generated elements fill large gaps between previous elements.

![ed8c9108f4764b149565f977e034617fa4e5b81760ab4b2f8d2c9f6795bc4129](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/42677e6a-de7c-4b39-b95a-305f4a893a21)

To account for the K-fold CV secondary sampling bias due to random sequences, we generate a 16-element artificial dataset containing four binary features and a label vector with 16 categories (i.e. one category for each combination of four binary eigenvalues). This dataset was replicated five times to form a complete set of 80 instances. For k-fold CV, different random seeds were used to repeat k from 2 to 10 for 50 times, and the error rate was extremely high, and the error rate decreased when k increased, as shown in Figure 3. It can also be seen from the graph that the results vary widely. So the traditional approach is obviously flawed, mainly because of the clumps and voids shown in Figure 1.

![f29a071e32f34c028664d0ac49f6477a9084cb6f292347ddbfe86b84fbafa5ce](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/e74b13f4-24c5-4709-9629-29317fd6c71d)

### Model effect test (MASE RMSSE)

After studying the content of the above paper and combining with the testing methods proposed by the teacher in class, we decided to optimize the time series based on the conventional testing methods such as MSE, and adopted the RMSSE MASE evaluation models commonly used by teachers for evaluation:

![image](https://github.com/supersteve2001/Financial-analysis-based-on-machine-learning-and-time-series-algorithms/assets/69947525/e67096de-88ee-44bd-af80-533566de704f)

In summary, deep forest and random forest have the best regression effect and obvious advantages in all models. Combined with the prediction in this paper, under different data types, both deep forest and random forest are likely to perform better. In terms of time series algorithms, due to the large number of factors affecting stock prices, the prediction effect of various models is average. However, it can be clearly seen that the ARIMA model optimized by SVM model has better performance in the two test quantities. However, there is still a gap between the general situation and deep forest, random forest and extreme forest.


