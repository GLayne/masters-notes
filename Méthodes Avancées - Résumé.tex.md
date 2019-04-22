# Méthodes Avancées en Exploitation de Données

#1 Introduction
##1.2 Basic Concepts

**Expected Prediction Error** **(p.11)**

$EPE(x) = E[(Y-\hat{g}(x))^2]$

$EPE(x) = E[(Y-g(x))^2] + (E[\hat{g}(x)] - g(x))^2 + E[(\hat{g}(x)-E[\hat{g}(x)])^2]$

$EPE(x) = Var(Y)+Bias^2+Var[\hat{g}(x)]$

$Var(Y)=\text{Error or a new observation that we cannot predict.}$

$Bias^2+Var[\hat{g}(x)]=\text{Bias-Variance trade-off where:}$

​	$Bias^2=\text{Difference between the true mean and the expected value of the prediction.}$

​	$Var[\hat{g}(x)]=\text{Variance of the prediction.}$

<u>Finding the best predictive model amounts to minizing the sum of the last two terms: a.k.a. find a good comproimse between the bias and the variance.</u>

*Explanatory modeling*: We focus on having a small bias in order to get a model close to the true model.

 *Predictive modeling*: A true model with large variance will perform poorly compared to a balanced one, even if it is the true one.

The predictive value of a variable depends not only on its individual effect, but also on its correlation with other predictors.



##1.3 Model Selection: Sample Splitting, Cross-Validation, AIC, BIC

###1.3.1 Maximum Likelihood Estimation (p.16)

The ML method considers the data fixed and look at it as a function of the parameters. This is the likelihood function. The idea is then to maximize the resulting function with respect to the parameters. Maximizing the log of the function is equivalent.

The MLE estimators of the regression parameters in a basic linear regression model are the same as the LS estimators.

Supposing we have $n$ independent observations, the MLE of $\theta_{mle}$ has the following properties:

1. $\hat{\theta}_{mle}$ is convergent, that is $\hat{\theta}_{mle} \rightarrow \theta$, as $n \rightarrow \infty$
2. $\hat{\theta}_{mle}$ is asymptotically normally distributed.
3. $\hat{\theta}_{mle}$ has the smallest variance among all estimators that are convergent and asymptotically normally distributed. (This is why $\hat{\theta}_{mle}$ is the best estimator among all possible estimators for parametric models).

###1.3.2 Likelihood Based Criteria (p.17)

**Problem**: The $-2LL(\hat\theta_{mle})$ measure can only decrease or stay the same as we add more variables.

**Solution:**

- **AIC**: $-2LL(\hat\theta_{mle}) +2p$

- **BIC**: $-2LL(\hat\theta_{mle}) +log(n)p$

The penalty of the BIC is always greater than the one from AIC, at least as soon as n > 7, which will always be the case.

###1.3.3 Sample Splitting Methods (p.12)

MLE is not the basis of all models, this is why we need general selection methods, as we can't use AIC and BIC everywhere.

**Train-Validation-Split**

If we need to obtain a valid estimation of the error: **Train-Validation-Test Split**

After selecting the final model, we should fit the model with all of the data.

###1.3.4 Cross-Validation (p.21)

**K-Fold Cross-Validation**:

Divide the dataset into K folds, for each fold:

- Use the data minus the fold ($K-1$ folds) to train the model
- Use the removed fold to test the model.

###1.3.5 $1-SE$ rule With Cross-Validation (p.21-22)

The $1-SE$ rule attempts to go to models that are simpler while still remaining close from the best models.

The $1-SE$ rule selects the model with the largest value of $\theta$ such that:

$CV(\theta)\le CV(\hat\theta_{min}) + SE(\hat\theta_{min})$

where $\theta$ is the value of a tuning parameter of a model (e.g. the $\lambda$ in the lasso or $n - k$ where k is the number of variables in the model).

where $\hat\theta_{min} = argmin\ CV(\theta),\ \theta \isin\{\theta_1, \theta_2,...,\theta_M\}$

**In summary, we want the simplest model with an error within 1 standard-error of the error of the model with the smallest error.**

To get Standard Errors:

$SE(\theta)={\sqrt{Var(CV_1(\theta),CV_2(\theta),...,CV_K(\theta))}\over\sqrt{K}}=\sqrt{{\sum^K_{k=1}{(CV(\theta_k)-\bar{CV})^2}\over K(K-1)}}$

where $\bar{CV}= {\sum^K_{k=1}{CV(\theta_k)}\over K}$ (average of Cross-Validated $\theta$ values)



##1.4 Generalized Linear Models (p.23)

Encompasses **Linear**, **Poisson** and **Logistic** Regressions.

The goal is to model $Y$ as a function of $X$: $\mu = E[Y|X]$, the conditional mean of Y given the covariates.

Two components are required to define a GLM:

1. An **Assumed Probability** Distribution of $Y$
2. A **link function**: continuous function $g$ linking the conditional mean to the covariates via $g(\mu)=\beta_0 +\beta_1X_1+...+\beta_pX_p$.

The conditional variance of $Y$ given the covariates is then given by:

$Var[Y|X] = \phi v(\mu)$, where $v$ is a known function that depends on the probability distribution.

The conditional variance is a function of the conditional mean multiplied by an over-dispersion parameter, estimated with the maximum likelihood method.



#### Case 1: Linear Regression under normality (normal distribution) (p.24)

For the normal distribution, the variance is not linked to the mean, hence $v(\mu)$ is a constant and we can take it to be 1.

So we have:

$g(\mu)=\mu=E[Y|X]=\beta_0 +\beta_1X_1+...+\beta_pX_p$

$Var[Y|X] = \phi$

If we rename $\phi$ to $\sigma^2$, we see that the GLM is the same as the ordinary linear regression model, if we assume that Y is from the $N(\beta_0 +\beta_1X_1+...+\beta_pX_p, \sigma^2)$ distribution.

#### Case 2: Logistic Regression (p.24-25)

If $Y$ is binary:

$E[Y|X] = P(Y=1|X)$

$Var[Y|X] = E[Y|X](1-E[Y|X]) = P(Y=1|X)(1-P(Y=1|X))$

Specifying a model for $E[Y|X]$ amounts to specifying one for $P(Y=1|X)$.

**Link function: logit:**  $g(\mu)=log({\mu\over 1-\mu})$

Logistic Regression Model is thus (after simplification):

$P(Y=1|X) = {1\over 1+exp(-(\beta_0 +\beta_1X_1+...+\beta_pX_p))}$



#### Case 3: Poisson Regression (p.25)

Parameter $\lambda >0$

$E[X] = Var[X] = \lambda$

**Link function: log**: $g(\mu) = log(\mu)$

Poisson regression model is thus:

$log(E[Y|X])=\beta_0 +\beta_1X_1+...+\beta_pX_p$

or equivalently:

$E[Y|X] = exp(\beta_0 +\beta_1X_1+...+\beta_pX_p)$

#2 Regularization and Variable Selection in Regression Models (p.27)
##2.1 Introduction (p.27-28)

**Goal**: Select a good subset of covariates, depending no the goal of the analysis (prediction vs. inference)

**From p.42: Regularization methods are ways to promote simpler models to try to avoid over-fitting.** Very important when the number of free parameters is large compared to the sample size.

Forward, Backward and Stepwise perform **hard** **variable** **selection**: covariates are in or out.

**Ridge regression** is not a **variable selection** method because all covariates remain in the model, but it does **shrinkage** (reduces the amplitude of the $\beta$).

p.45: **Lasso** performs **shrinkage** and **variable selection** at the same time.

##2.2 Classical Variable Selection Methods (p.28)

###2.2.1 Forward, Backward, Stepwise, and All Subset Selection (p.28)

#### Forward (p.28-29)

1. Add each variable in $S_{out}$, one at a time, to the model that contains the variables in $S_{in}$ and compute the entry criterion for each of them.

2. Select the variable with the best value of the entry criterion, but only if it satisfies the entry condition. If no variable satisfy the entry condition, then the process stops and the selected covariates are the ones in $S_{in}$. 

3. If the entry condition is satisfied, add the best variable to $S_{in}$ and remove it from $S_{out}$. 

4. Repeat 1 to 3. 

   *With the forward method, when a variable is entered, it stays in the model. The classical entry criterion is the p-value of the added variable.*

#### Backward (p.29)

Same as Forward, but in the opposite direction. Start with a full model and remove all variables one by one, and select for permanent removal the variable which has the bst value of the exit criterion, but only if it satisfies the threshold.

*With the backward method, when a variable is removed, it cannot go back in the model later.*

#### Stepwise (p.30)

Combines Forward and Backward:

1. Perform a forward step. 
2. Perform backward steps as long as variables are removed. 
3. If no variable was entered and no variable was removed, then stop. The selected covariates are the ones in $S_{in}$. 
4. If one variable was entered and/or at least one variable was removed, then go back to 1. 

*With the stepwise method, a variable can enter the model and be removed later in the process. This can be useful when the covariates are correlated.*

#### All-Subset (p.30)

This method evaluates all possible models and select the one with the best value of a criterion, for
example the AIC or BIC. But this is very computer intensive, as there are $2^p$ possible models for $p$ covariates.

###2.2.2 Ridge Regression

#### Multicollinearity (p.31-32)

<u>Multicollinearity is when some linear combinations of covariates are very correlated with other linear combinations of covariates.</u> 

It can potentially make the estimates of some of the $\beta$ very unstable (small changes in the data can cause great changes in the values of these estimates). 

More precisely, the **global effect of a group of highly correlated covariates will still be accounted for by the model, but the allocation of this global effect across the covariates will be difficult to make.**

#### Ridge Regression (p.33-34)

The Ridge Regression minimizes this "penalized" cost function:

$\sum_{i=1}^n{(Y_i-(\beta_0+\beta_1X_{1i}+\beta_2X_{2i}+...+\beta_pX_{pi}))^2}+\lambda\sum_{j=1}^p\beta^2_j$

where $\lambda \ge 0$ is a complexity parameter.

Generally, we standardize the covariates before estimating the $\beta$ parameters, so that the penalty affects the estimation of each covariate in the same way. 

Again (p.43): **Ridge regression** is not a variable selection method because all covariates remain in the model, but it does **shrinkage** (reduces the amplitude of the $\beta$).

###2.2.3 Simple Example

Skipped.

##2.3 Lasso and Other Methods (p.43)

###2.3.1 The Lasso (p.43-45)

The Lasso estimates minimize:

$\sum_{i=1}^n{(Y_i-(\beta_0+\beta_1X_{1i}+\beta_2X_{2i}+...+\beta_pX_{pi}))^2}+\lambda\sum_{j=1}^p|\beta_j|$

Same as the Ridge Regression, except for the penalty, which is the sum of the absolute values of the $\beta$ parameters.

**As $\lambda$ increases, some of the coefficients will become exactly 0.**

**Lasso performs shrinkage and variable selection at the same time.**

The covariates are usually standardized before estimating the coefficients.

<u>Biggest advantage</u>: achieving good predictive performance with a simpler model (less variables).

#### Effects on strongly correlated covariates (p.47)

- **Ridge regression** has the tendency to shrink them together to get some kind of average effect.

- **Lasso regression** will tend to favor one of the correlated covariates.

###2.3.2 The Elastic Net (p.48)

Combines both penalties.

The Elastic Net minimizes the following cost function:

$min_{\beta_0,...,\beta_p}{1\over n}\sum_{i=1}^{2n}{(Y_i-(\beta_0+\beta_1X_{1i}+\beta_2X_{2i}+...+\beta_pX_{pi}))^2}+\lambda[(1-\alpha)/2\sum_{j=1}^p(\beta_j)^2+\alpha\sum_{j=1}^p|\beta_j|]$

$\lambda$ is the shrinkage parameter.

$\alpha$ is a weighting factor between the two penalties: Ridge is $\alpha = 0$ ; Lasso is $\alpha = 1$.

###2.3.3 Relaxed Lasso (p.50)

The Lasso estimates are biased towards 0. When there is a large number of covariates, a large penalty may be needed to perform good variable selection, but that will also affect the estimates of the remaining covariates.

A solution is to use the lasso as a variable selection tool and then perform ordinary least square regression on the selected covariates (without any regularization).

We could also re-run a lasso on the selected covariates, but, the second time, the $\lambda$ paramter should be kept small.

##2.4 An Example: Ames Data

Skipped.

##2.5 Grouped Variables (p.78)

The **grouped lasso** will enforce group selection: either all variables within a group are selected, or they are all dropped.

Other methods perform **bi-level selection**, which will try to select important groups and variables within groups, selecting subsets of variables within groups.

The **grouped lasso estimate** minimizes: 

$\sum_{i=1}^n{(Y_i-(\beta_0+\beta_1X_{1i}+\beta_2X_{2i}+...+\beta_pX_{pi}))^2}+\lambda\sum_{l=1}^L{\sqrt {p_l}}\ (\sum_{j\isin G_l}\beta^2_j)^{1/2}$

where the $p$ covariates are partitioned into $L$ groups: $G_1,G_2,...,G_L$ with sizes $p_1, p_2, ...,p_L$.

p.89: *For predictive purposes, these methods do not generally have advantages over ones that treat the dummies as individual variables regardless of the original covariates. But they might be helpful if the goal is variable selection.*



###2.5.1 Ames Data Example (continued)

Skipped.

##2.6 Using Interactions (p.89)

#### Hierarchy 

The concept of hierarchy is that if an interaction between variables is present in the model, then all lower order interactions between these variables should also be present.

**Strong hierarchy:** If $X_1 * X_2$  is in the model, then <u>both</u> stand-alone variables should be in the model.

**Weak hierarchy:** If $X_1 * X_2$  is in the model, then <u>at least one</u> stand-alone variable should be in the model.

<u>Most statisticians think that strong hierarchy should be enforced.</u>

**However, it is not clear that enforcing a hierarchy principle will improve prediction results.** 




##2.7 Sure Independence Screening (SIS) (p.91)

$p>>n$ case: case when the number of predictors $p$ is a lot larger than the number of observations $n$.

**Problem**: Performance of variable selection methods is adversely affected and the computational cost to apply some of them may be too high.

The goal is to quickly reduce the number of predictors to a more manageable size (usually below sample size) without losing anything important.

Then, another variable selection method can be used on the remaining variables.

Process:

1. Compute the correlation between each predictor and the reponse $Y$.
2. Rank results according to the magnitude in absolute value of the correlation.
3. Keep the top $n-1$ or $n/log(n)$ variables with highest correlation.
4. 

##2.8 Generalized Linear Models (p.91-92)

For cases other than linear regression, the regression method can be adapted. As an example, the elastic net: (note the addition of $w_il$, the first sum is computed up to $n$ and not $2n$, and the $l$ is the negative log-likelihood contribution for observation $i$)

$min_{\beta_0,...,\beta_p}{1\over n}\sum_{i=1}^{n}w_il{(Y_i, \beta_0+\beta_1X_{1i}+\beta_2X_{2i}+...+\beta_pX_{pi})^2}+\lambda[(1-\alpha)/2\sum_{j=1}^p(\beta_j)^2+\alpha\sum_{j=1}^p|\beta_j|]$



##2.9 German Credit Data

Skipped, for the most part.

#### ROC Curve & AUC (p.105)

Y Axis: True positive rate = $\text{Sensitivity}$ = $P(\hat Y=1|Y=1)$

X Axis: False positive rate = $1- \text{Specificity}$ = $P(\hat Y=1|Y=0)$

The more the ROC curve is high in the upper left quadrant, the better the model.

We measure this with the **AUC** (Area Under the ROC Curve).

#### Lift Chart (p.106)

We plot the **true positive rate** in the Y axis and the **rate of positive predictions** in the X axis:

- True Positive Rate: % of predicted Y=1 obtained by the model's predictions

- Rate of positive predictions: We order our predicted probabilities of Y=1 descending and we declare the top Z% as 1 naively.

  By comparing these two rates, we know how much better is our model at predicting Y=1, compared to a naive rule.

###2.9.1 AUC as a Concordance Index (p.108)

The AUC can be seen as the C-Index.

Considering all pairs of observations $(i,j)$. A pair is valid if $y_i ≠ y_j$. Among the valid pairs, a pair is called:

- concordant if ($y_i > y_j$ and $\hat p_i >\hat p_j$) or if ($y_i < y_j$ and $\hat p_i <\hat p_j$)
- discordant if ($y_i > y_j$ and $\hat p_i < \hat p_j$) or if ($y_i < y_j$ and $\hat p_i > \hat p_j$)
- tied if $\hat p_i = \hat p_j$.

A **concordant** pair is one for which the predicted probabilities and the true observations are in the same order. 

A **discordant** pair is one for which the predicted probabilities and the true observations are in the reverse order.

**C-Index:** the proportion of concording pairs: Concordant pairs receive a weight of 1, Discordant : 0 and Tied: 0.5. The C-Index is the sum of thse weights over all valid pairs divided by the number of valid pairs.

##2.10 Other Applications of Regularization (p.110)

Examples: 

- Weight decay in Neural Networks

- As part of PCA (p.110 to p.116)

#3 Tree-Based Methods and Random Forests (RF) (p.117)
##3.1 Introduction to Tree-Based Methods (p.117)

Data-driven partition of the covariates space that locally fits the data.

Model: $\hat f(X)=\sum_{j=1}^md_jI(X\isin R_j)$

where $I(X\isin R_j)=1$ if the condition is true and 0 otherwise (i.e. the data point belongs in the region $R_j$).

#### Trees have many advantages (taken from p.129):

They can handle any types of covariates, can model any types of target $Y$, can detect certain types of interactions automatically, scale well to large samples sizes, and small trees are easy to interpret.

#### Trees also have disadvantages (taken from p.129):

They can often be beaten by other methods in terms of prediction performance and the interpretability of trees is quickly lost when the tree is large.

###3.1.1 Classification and Regression Trees (CART) (p.118)

#### Building the Tree

We split recursively using only the data which is in the region of the parent node.

CART evaluates all possible splits on all covariations and selects the one that has the best value of a criterion.

For a regression tree, the criterion is the squared-error. 

For a classification tree, the criterion can be: the Gini index, or others...

#### Pruning the Tree (p.119)

Large trees will usually overfit the data. Pruning will reduce this. 

CART uses cost-complexity pruning:

$C_\alpha(T)= SSE(T)+\alpha N_T$ 

where $SSE(T)$ is the in-sample sum of squares of the error when we predict each training data point by the tree prediction and $N_T$ is the number of terminal nodes.

$SSE(T)$ is the apparent error (in-sample error)

$N_T$ measures the complexity of the model

$\alpha$ is the tradeoff between tree complexity and goodness of fit

#### Gini Index for Classification Trees (p.120)

Gini Index: In a node $t$, let $\hat p_{tk}$ be the proportion of observations with a response value $k$.

$G(t) = \sum_{k=1}^K \hat p_{tk}(1-\hat p_{tk})$

Small values of G indicates that the $\hat p_{tk}$ are more dispersed, that is that the node contains predominantly observations from a single class.

The best split is the one that minimizes the weighted sum of the Gini index:

$n_L G(t_L) + n_R G(t_R)$

The prediction for classification trees is the majority class.



#### Advantages of the CART method compared to parametric models (p.120)

- If we apply any transformation that preserves the order of a continuous or ordinal predictor, then using the transformed variable or the original one will produce the same tree.
- A tree can automatically detect certain types of interactions between the predictors, so no need to create interaction variables.

###3.1.2 Conditional Inference Trees (p.121)

#### Disadvantages of the CART method

CART trees can suffer from biased split selection: CART tends to favor splitting on variables that have a lot of possible splits.

To resolve this, we need to perform the variable selection and best split selection in separate steps (which is done at the same step in CART).

To do so:

At each node, perform a global null hypothesis between all covariates and the response $Y$:

$H_0$ : all $\beta = 0$

$H_1:$ at least one $\beta \ne 0$  

If we cannot reject $H_0$, we create a terminal node at that point. Otherwise, select the covariate $X_j$ with the strongest association to $Y$ and then split the node with $X_j$ according to a criterion.

This method requires **no pruning** because the tree is built in a forward fashion.

**Details on how to measure the association with $Y$ are laid out at page 122.**

*In contrary to CART, this method is <u>not</u> invariant to monotone transformations!*

##3.2 Example: Breast Cancer Data

Skipped

##3.3 Basic Random Forest (p.129)

Combining many trees can improve drastically the prediction performance of a single tree. Ensembles of trees are difficult to interpet but a large tree is also difficult to interpret, so it doesn't really matter.

Trees are unstable learners, a small modification in the dataset can produce a very different tree (small bias, large variance).

#### Original random forest algorithm (p.130)

With $p$ covariates, select the number of trees $B$, and the number of covariates $p_0 ≤ p$, to select at random at a node to find the best split. For $b=1 $ to $B$: 

1. Create a bootstrap sample from the original data. 
2. Build a tree with the bootstrap sample (large trees are usually built and no pruning is performed). At each node, select at random $p_0$ out of the $p$ covariates and find the best split with these covariates only. Note that the subset of covariates can vary from node to node. Let $\hat T(x)$ be this tree. 

The final prediction model is :

For continuous $Y$, get the average prediction of $Y$ for all trees.

For categorical $Y$, either: get the majority class from each tree *or* get the average of probabilities for each class from each tree and select the class with the highest averaged proportion.

#### Two sources of randomness

- Using a new bootstrap sample for each tree;
- Selecting a subset of covariates at each node.

Default values of $p_0$: usually $p/3$ for regression and $\sqrt p$ for classification forests. 

Very often, using $p_0 < p$ produces better results, because resulting trees are less correlated and are able to find more structure in the data.



#### Out of Bag Observatios (OOB) (p.131)

For a given tree, the OOB observations are those that are not part of the bootstrap sample (and not used to build the tree). On average, it amounts to about 37% of the observations and this can be used as test data.

We can get OOB predictions for each tree, which allows us to get valid predictions or optimize parameters (like $p_0$) without resorting to cross-validation.

The OOB prediction for a given observation is the average of the predictions for all trees where this observations was part of the OOB set.

##3.4 Implementations of the Original Random Forest Algorithm

Skipped

##3.5 Forest with Other Tree Building Algorithms

Skipped

##3.6 Ames Data (continued)

Skipped

##3.7 German Credit (continued)

Skipped

##3.8 Nearest Neighbour Forest Weights (p.143)

The Random Forest prediction is a weighted average of the training observations.

The Bag of Observations for Prediction (BOP) is the pooled set of training observations that are in the same terminal nodes as the new data point ($x_{new}$) for the whole forest.

The idea is to use $BOP(x_{new})$ to compute any desired summary for this new observation.

##3.9 Variable Importance with Random Forests (p.148)

Ways to evaluate the importance of the covariates within the RF models.

We can use them to perform variable selection with a random forest.

#### Original VIMP

Idea: $X$ is important for predicting $Y$ if the prediction error increases when the link between $X$ and $Y$ is broken.

For each tree, we use the OOB samples to calculate the error using the original OOB sample and the error using the OOB sample for the same tree, but with permuted order for the variable $X$, that is. If there is a big difference in the error due to the permutation, then the ordering of $X$ (its structure) was important in predicting $Y$.

A permutation looks like this:

Original:									Permuted:

X: 2.3   4.5   6.7   8.1				X: **4.5   8.1   2.3  6.7**

Y:  1.2  2.5  4.6  7.3				 Y: 1.2   2.5   4.6  7.3	

The final VIMP is the average of these over the trees.

Formal definitions page 149.



#4 Boosting (p.153)

Boosting algorithms take weak learning algorithms and boost it into a strong one.

##4.1 Adaboost (p.153)

Example with binary classifier

Target Class $Y \isin \{-1, 1\}$

Initialize observation weights $w_i=1/n$ for $i=1,...,n$. 

Select number of iterations $M$.

For $m=1$ to $M$:

1. Fit the classifier G to the training sample with observations weights $w_i$

   Let $\hat G_m$ be the fitted function

2. Compute the weighted prediction error $e_m = {\sum^n_{i=1}{w_iI(y_i \ne \hat G_m (x_i))}\over \sum^n_{i=1}{w_i}}$

3. Compute $\alpha_m = log[(1-e_m)/e_m]$

4. Update the weights by $w_i = w_i exp(\alpha_m I(y_i \ne \hat G_m (x_i))), i=1,...,n$

   $exp(0) = 1 \rarr$ observation weight doesn't change when $I(y_i \ne \hat G_m (x_i)) = 0$

   $exp(\alpha_m \cdot 1) \rarr$ observation weight changes when $I(y_i \ne \hat G_m (x_i)) = 1$: it is multiplied by $exp(\alpha_m)$

The final prediction model is $\hat G(x) = sign[\sum_{m=1}^M \alpha_m \hat G_m (x)]$



##### Two main ideas:

1. A combination of many classifiers are used as the final model.
2. At a given iteration, observations that were misclassified in the previous iteration are up-weighted and the ones that were well classified keep the same weight.

We want to force the classifier to adapt to the misclassified observations.

In the end, the classifier is a weighted average of each individual classifier with weight ($\alpha_m$) that depends on its error. The smaller the error $e_m$ the larget the weight $\alpha_m$.

##4.2 German Credit Example (continued)

Passed.

##4.3 Boosting with the Least-Squares Criterion (p.157-158)

Select a shrinkage factor $\epsilon > 0 $, number of iterations $M$, and learner $g$

Initialize fit to $\hat G_0 = 0$ and residual vector $r = (r_1,...,r_n)^\prime= y = (y_1,...,y_n)^\prime$, the values of $Y$ in the sample. For $m = 1$ to $M$:

1. Fit the learner to the current residuals as the target with the least-squares criterion. That is, find $\hat g_m$ that minimizes

   $\sum_{i=1}^n {(r_i - g(x_i))^2}$

2. Update $\hat G_m(x) = \hat G_{m-1}+\epsilon \hat g_m(x)$

   where $\epsilon$ is the shrinkage factor, 

   The final model is $\hat G_M(x)$ (the last learner (**not** the average of all models))

**How it works**

At each step, the learner is fit to the current residuals (the <u>unexplained</u> part of $Y$ at this point).

We then update the current learner but only by a small amount controlled by $\epsilon$.

Thus, $\hat G_m(x)$ is updated with a shrunken version of the function that best fits the current residual.

##### Component-wise Method (p.158)

- It is not clear that solving the least-squares problem in step 1 is easy (it depends on the selected learner $g$). 
- One easy way is to use a linear model and a variable selection method. At each iteration $m$, select the variable that will minimize the criterion (that is, we select the best variable amon gthe p variables at each step). In the end, the final learner $\hat G_m(x)$ will be a linear combination of the covariates. It is likely that some of the covariates will not have been selected at all.

Typically, we use trees as the base learner for g, because it can be better to let the data find the effects for us.

##4.4 Ames Data example (continued)

Passed.

##4.5 Boosting with a General Loss Function (p.160-161)

Select a shrinkage factor $\epsilon > 0 $, number of iterations $M$, learner $g$ **and loss function $L$.**

Initialize fit to $\hat G_0 = 0$. For $m = 1$ to $M$:

1. Find $\hat g_m$ that minimizes:

   $\sum_{i=1}^n L(y_i, \hat G_{m-1}(x_i) + g_m(x_i))$

2. Update $\hat G_m(x) = \hat G_{m-1}+\epsilon \hat g_m(x)$

   where $\epsilon$ is the shrinkage factor, 

   The final model is $\hat G_M(x)$ (the last learner (**not** the average of all models))

$L$ represents a loss function, for example : $L_2 : (y-\hat y)^2$; $|y-\hat y|$  ; $-2LL$ ; $exp(-yg(x))$ (Adaboost), etc.

Solving step 1 is not always easy. Gradient boosting provides an approximation of that.

##4.6 Gradient Boosting (p.162)

#### Gradient Descent

Lets us find the minimum of a function.

##### One-dimensional gradient descent:

At the $i^{th}$ iteration, update $x$ with:  $x^{(i)}=x^{i-1} - \epsilon f^\prime (x^{i-1})$ 

If $f^\prime >0 \rarr$ go left ;  if $f^\prime < 0 \rarr$ go right

Stop when $|x^{(i)} - x^{(i-1)}|$ is small enough. The minimum of the function is the last value of $x$.

**Multi-dimensional gradient descent:**

Same as above, but we take a step in the right direction for all arugments (all dimensions) in a given iteration. We do so by looking at the gradient of all variables at the same time.

#### Gradient Boosting

Gradient boosting uses this method by viewing the loss as the function to be minimized and each data point as an argument (variable). Instead of having $q$ arguments, we have $ntrain$ arguments.

##### Algorithm

Select a shrinkage factor $\epsilon > 0 $, number of iterations $M$, learner $g$ and loss function $L$.

Initialize fit to $\hat G_0 = 0$. 

For $m = 1$ to $M$:

1. For $i=1,...,n$, (for each observation) compute the pointwise negative gradient of the loss function at the current model:

   $r_i = {\partial L(y_i, \hat G_{m-1}(x_i)) \over \partial \hat G_{m-1}(x_i) }$

2. Fit the learner with the $r_i$s as the target with the least squares criterion. That is, find $\hat g_m$ that minimizes:

   $\sum_{i=1}^n{(r_i-g(x_i))^2}$

3. Update $\hat G_m (x)= \hat G_{m-1}(x)+\epsilon \hat g_m(x)$

The final prediction model is $\hat G_M(x)$    (the model from the last iteration).

**In summary**: 

Step 1) we compute pointwise gradients;

Step 2) We want the learner to move in the direction of the gradient for all points, so we learn to predict the gradient using pointwise gradients as the training data;

Step 3) We move the global learner in the right direction.

##4.7 Ames Data example (continued)

Passed.

#5 Survival Analysis (p.171)

##5.1 Basic Concepts in Survival Analysis

**Target Variable**: The time before an event of interest occurs.

#### $S(t)$ : **Survival Function of T** : 

$S(t) = P(T>t) = 1 - F(t)$

where $F(t)$ is the cumulative distribution function (cdf) of T, that is $F(t) = P(T \le t)$

#### $h(t)$ : **Hazard Function of T** :

- Discrete $T$ : $h(t) = P(T=t|T\ge t)$
  - Represents the probability that an event happens at time $t$ given the fact that it hasn't happened yet.
- Continuous $T$ : $h(t) ={ f(t) \over S(t)}$  where $f$ is the density function of $T$
  - Does not represent a probability because it can take values greater than 1, but is always $\ge 0$. Represents the instantaneous risk of experiencing the event at time $t$ given it did not occur before.

#### $H(t)$ : Cumulative Hazard Function (or Integrated Hazard Function) (p.172)

$H(t)=\int_0^t {h(u)du} $    : represents the accumulated risk up to time $t$.

The greater $H(t)$ is, the greater the risk that the event occurs by time $t$.

#### Finding the missing function (p.172)

Once we know one function, we can know the others:

$h(t) = - {d \over dt} log(S(t))$     and     $S(t) = exp(- \int_0^t h(u)du) = exp(-H(t))$

#### Censoring

**Right-Censoring:** When at the end of the study, we still don't know when the event occurs for some subjects.

#### Notation

The observed data is represented by: 

1. $\tau = min(T, C)$    : what we observed in reality ($\tau$ will be $C$ if censored and $T$ if not censored)

2. $\delta =I(T \le C)$     ($\delta = 1$ if  we observed the event (not censored);    $\delta = 0$ if we have a censored time).

3. $X_1, X_2, ..., X_p$

#### Assumptions

We assume that $T$ and $C$ are conditionally independent given the covariates.

In the base model, the covariates are assumed as time-independent (they do not vary with time).

####  Making Predictions : Mean Survival Time (p.174)

For a positive random variable of survival time $T$, we have $E[T]=\int_0^\infty S(t)dt$

where $S$ is the survival function of $T.$ This allows us to compute the **mean survival time of a subject.**

#### Restricted Mean Survival Time (p.174)

$RMST(t_M) = E[min(T, t_m)] = \int_0^{t_M} S(t)dt$

We limit the integral's maximal time value by $t_m$ in order to calculate $E[T]$. 

By doing so, we limit the impact of having to suppose a model for high values of $T$ (based on the fact that we have a lot of censored data and that the distribution at the right end of the distribution of T is imprecise.) 

##5.2 Cox Model (p.175)

Semiparametric model which specifies how the covariates modify the hazard function (without it being fully specified).

$h(t|x_1, x_2,...,x_p)$ denotes the risk function for a subject.

The Cox Model is as follows:

$h(t|x_1, x_2, ..., x_p) = h_0(t)exp(\beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p )$

	- $h_0(t)$ is the base risk function at origin (non-parametric)

  - the rest is the effects of the covariates (parametric)

#### Interpretation of effects

Increasing $x_j$ by 1 while the other covariates remain fixed multiplies the risk by $exp(\beta_j)$

(it multiplies the whole $h_0(t)$ function)

#### Partial Likelihood Function (p.176)

Since the model is semi-parametric, **we cannot use MLE to estimate parameters**, we use a **partial likelihood function: **

- We only use times where events happened
- We compare the risk for an individual at the event time against the risk of others for which the event did not happen yet.

Thus, we never predict $T$, but we use the order between subjects. That is, we model the probability that an event happens to someone, compared to it happening to all other remaining people (which haven't experienced the event). **(p. 181)**

###5.2.1 Simple Numeric Example With the Partial Likelihood Function 

Passed.

##5.3 Churn Articial Data Example

Passed.

##5.4 Fully Parametric Models (p.194)

One type of fully parametric models: **Accelerated Failure Time (AFT) Models:**

​		$T= \phi(x,\beta) T_0$

​		where: $T_0 = $ time of an individual with $\O$ covariates ("individual at origin")

​					 $\phi$ is a positive function linking the covariates $x$ to an unknown vector of parameters $\beta$		

**Intuition:** Basically, the covariates accelerate or decelerate, through $,\phi(x,\beta)$ the survival time of a subject compared to the reference subject.

For example, if $\phi(x,\beta) = 2$ then the survival time of that subject is on average twice as long as the one from the reference subject.

##### Function $\phi$

Can be specified in many ways, but the most common is:   $\phi(x,\beta) = exp(\beta_1x_1+\beta_2x_2+...+\beta_px_p)$

The model becomes: $T=exp(\beta_1x_1+\beta_2x_2+...+\beta_px_p)T_0$

##### Interpretation

This means that when $x_j$ increases by 1 (and all other variables remain unchanged) then the average survival time is multiplied by $exp(\beta_j)$.

<u>The difference with the Cox Model:</u> The effects are multiplicative <u>on the average survival time and **not** the risk.</u>

#### Predictions using the AFT model  (p.195-196)

Can provide predictions of survival time for new subjects at time 0 but also for ongoing subjects (one still alive at time t*).

We usually want to estimate the residual lifetime: if $S_T$ is the survival function of $T$, we have:

$E[T|T>t^*] = t^* + {\int_{t^*}^\infty S_T(t)dt \over S_T (t^*)}$

​		where: - $E[T|T>t^*]$ is the average residual lifetime that we want to predict

​					   - $t^*$ is the observed time

​						- the integral over $S_T(t^*)$ is the estimated residual time.



###5.4.1 Simple Numeric Example With the Partial Likelihood Function (continued)

Passed.

##5.5 Details About the function survreg

Passed.

##5.6 Churn Articial Data Example (continued)

Passed.

##5.7 Survival Trees and Forests (p.207)

One of the most popular splitting rule for survival data is the **log-rank test** where we compare the survival functions for different groups.

Details available at page 207

##5.8 Churn Articial Data Example (continued) 

Passed.

##5.9 Evaluating a Model With Survival Data (p.210)

Problem: since we have censored data, we don't know the true survival time for some of the observations in the validation set and, as such, **it is impossible to compute the error of the model.**

**Solutions**: Brier Score and C-Index, among others

###5.9.1 Brier Score and Integrated Brier Score (p.211)

**Brier Score (general use - not for survival models)**

$BS = {1 \over n} \sum_{i=1}^n (y_i - \hat p (x_i))^2$

This is different from the misclassification rate: $MCR = {1 \over n} \sum_{i=1}^n (y_i - \hat y_i)^2$

##### Brier Score for Survival Models - without censoring: (p.212)

We have a binary target $I(T>t)$ and a probability that this target takes a value of 1: $\hat S(t|\bold x)$.

If there is no censoring, the Brier Score at time $t$ is : 

$BS = {1 \over n} \sum_{i=1}^n (I(\tau_i > t) - \hat S(t|\bold x_i))^2 $

that is the average difference between the binary <u>outcome</u> at time $t$ minus the <u>probability</u> that the target takes a value of 1.

####IBS

If we compute the integral of $BS(t)$ with respect to $t$ we can get a single value for the whole function (not limited to a specific time $t$):

$IBS ={ 1 \over max(\tau_i)} \int_0^{max(\tau_i)} BS(t)dt$   = average of BS across all times $t$. 

**Lower values of IBS indicate better performances.** The IBS is an integrated weighted squared distance between the estimated survival function and the empirical survival curve.

**see graph page 213** 

#### Brier Score with censoring (p.214)

When there is no censoring, we fall back to $BS = {1 \over n} \sum_{i=1}^n (I(\tau_i > t) - \hat S(t|\bold x_i))^2 $

$BS(t) = {1 \over n} \sum_{i=1}^n \Bigl((\hat S(t|\bold x_i)^2I(\tau_i \le t \and \delta_i =1)\hat G^{-1}(\tau_i)+(1-\hat S(t|\bold x_i))^2I(\tau_i > t)\hat G^{-1}(t)\Bigr)$

This means:

- **Scenario 1:** if $\tau_i \le t \and \delta_i =1$: the event occured before or at time $t$:
  - The contribution to $BS(t)$ is $(0-\hat S(t| \bold x_i))^2$
  - IPCW weight = $1/\hat G(\tau_i)$   (or $\hat G ^{-1}(\tau_i)$), where $G(\tau_i) = P(C_i > \tau_i|\bold x_i)$

- **Scenario 2:** if $\tau_i > t$: the event has not occured yet at time $t$ (here we don't care if it's censored or not):
  - The contribution to $BS(t)$ is $(1-\hat S(t|\bold x_i))^2$
  - IPCW weight = $1/\hat G(t)$   (or $\hat G ^{-1}(t)$), where $G(t)=P(C_i > t|\bold x_i)$
- **Scenario 3:** $\tau_i \le t \and \delta_i = 0$: the observation is censored before or at time $t$:
  - We don't know what is the contribution to $BS(t)$ so we don't use them (weight of 0).
  - The IPCW for scenario 1 and 2 are used to compensate the fact that these observations are not used.

IPCW weights the observations from scenario 1 or 2 depending on the probability of being a censored observation at the time of the measurement or the event, depending on the case.

###5.9.2 Concordance Index or C-Index, or Harrell's Index (p.218)

C-Index is a concordance measure that evaluates if the predictions from a model are ranked in the same way as the observed times.

Let $\hat H(t|\bold x)$ denote the estimated cumulative hazard function and $\hat H_i = \sum_{l=1}^m \hat H(t_l|\bold x_i)$ (sum of $\hat H$ for all times $l$ for person $i$).

The idea is that if subject $i$ had the event occur at time $\tau_i$, say for example it occurs <u>before</u> subject $j$ at time $\tau_j$ and its estimated cumulative hazard function $\hat H_i$ is higher than $\hat H_j$, then the model is accurate. If the risk is lower than $\hat H_j$, then the model was not accurate. 

**Intuition: If the risk is higher for a person, then the event should happen to sooner.**

1. if $\tau_i < \tau_j \and \hat H_i > \hat H_j \and \delta_i = 1$ : the pair is <u>usable</u> and <u>concordant</u>: **the model was accurate**. It receives a weight of **1**.
2. if $\tau_i > \tau_j \and \hat H_i < \hat H_j \and \delta_i=1$ : the pair is <u>usable</u> and <u>concordant</u>: **the model was accurate**. It receives a weight of **1**.
3. if $\tau_i < \tau_j \and \hat H_i < \hat H_j \and \delta_i=1$ : the pair is <u>usable</u> and <u>discordant</u>: **the model was NOT accurate**. It receives a weight of **0**.
4. if $\tau_i > \tau_j \and \hat H_i > \hat H_j \and \delta_i=1$ : the pair is <u>usable</u> and <u>discordant</u>: **the model was NOT accurate**. It receives a weight of **0**.
5. if $\tau_i < \tau_j \and \delta_i = 0$ : the pair is <u>not usable.</u>
6. if $\tau_i > \tau_j \and \delta_j= 0$ : the pair is <u>not usable.</u> (watch out, it is $\delta$ **j** this time)

<u>We also do not consider pairs with tied observed times and/or tied risk predictions.</u>

The C-Index is as follows: **It is the proportion of concordant pairs among all usable pairs.**

$CI = {\sum_{i<j} (I(\tau_i <\tau_j)I(\hat H_i> \hat H_j)\delta_i + I(\tau_i >\tau_j)I(\hat H_i< \hat H_j)\delta_j)\over \sum_{i<j} (I(\tau_i < \tau_j)\delta_i + I(\tau_i >\tau_j)\delta_j)}$

Dropping all non-usable pairs might <u>introduce a bias.</u> Some more sophisticated versions incorporate the IPCW to take care of that.

##5.10 Regularization Methods With Survival Data (p.223)

Passed.

###5.10.1 Churn Articial Data Example (continued)

Passed.

##5.11 Boosting With Survival Data (p.235)

Passed.

###5.11.1 Churn Articial Data Example (continued)

Passed.

##5.12 Discrete-Time Survival Analysis (p.238)

The difference with discrete time is that the hazard is a conditional probability:

$h(t)=P(T=t|T\ge t,x_1(t),x_2(t),...,x_p(t))$

###5.12.1 Discrete Time Proportional Odds Model (DTPO) 

It is a Proportional Odds model just like in logistic regression, one logit model for each time $t$:

$log \Bigl({h(K) \over 1-h(K)} \Bigr) = \alpha_K +\beta_1 x_1(K) + \beta_2 x_2 (K) +...+\beta_px_p(K)$

- The effects ($\beta$) of covariates are assumed to be constant through time.
- The values of $x_j$ <u>can change</u> through time.

##### With time-varying effects for some variables

$log \Bigl({h(K) \over 1-h(K)} \Bigr) = \alpha_1 D_1(t) +\alpha_2 D_2(t)+...+\alpha_K D_K (t)$

​						$ +\  \beta_{11} x_1(K) + \beta_{12} x_2 (K) +...+\beta_{1K} D_K(t) x_1(t)$

​						$+\ \beta_2x_2(t)+...+\beta_px_p(t)$

We now have one $\beta$ for $x_1$ for each $K$

Where $K$ are the parameters for variable $x_1$, one per period.

The **survival function** is now : $S(t) = S(t-1)(1-h(t)), \text{for } t=1,2,...,K$  with $S(0) = 1$

The **probability function** is now: $\pi(t) = S(t-1) - S(t), \text{for } t=1,2,...,K$

###5.12.2 Churn Articial Data Example (continued)

Passed.

###5.12.3 Trees and Forests for Discrete-Time Survival Data (p.256)

We can create trees and forests by splitting subjects on its different periods. We will thus automatically detect interactions between covariates and the periods, if it is important for prediction.

Requires the **ypp** (y-person-period) data format : one row for each person-period combinaison.

###5.12.4 Churn Articial Data Example (continued) 

Passed.

##5.13 Time-Varying Covariates, Time-Varying Effects, And Landmark Analysis (p.258)
###5.13.1 Time-Varying Covariates And Effects

We can use the DTPO model to do this. For the Cox Model, we have these options:

#### Cox Model with Time-Varying Covariates ($x$)

$h(t|x_1, x_2, ..., x_p) = h_0(t)exp(\beta_1 x_1(t) + \beta_2 x_2(t) + ... + \beta_p x_p(t) )$

- Each $x$ is subject to time $t$ : $x(t)$, that is, the value of $x$ changes with time. We can also make it so some $x$ can change with time while others do not.

#### Cox Model with Time-Varying Effects ($\beta$) (p.259)

$h(t|x_1, x_2, ..., x_p) = h_0(t)exp(\beta_{11} x_1+ \beta_{12}x_1t+\beta_2 x_2 + ... + \beta_p x_p )$ 

-  Above: $x_1$is subject to two effects, one that is an interaction with $t$ and one that is not.

  

**NEVER, EVER use future information with respect to the value $t$ when building a model.** We can only use covariate information up to time $t$.

###5.13.2 Landmark Analysis (Dynamic Prediction) (p.260)

**Idea**: build models (usually Cox), at different landmark times $t$, using the covariates information available up to $t$ and only the subjects **still at risk**.

We still need to try to use information from other periods to reduce variability in the estimates, so 2 main approaches were tested:

- Approach 1: Stacking the data sets from many landmark times and fitting a single model
- Approach 2: Estimate the survival probabilities through joint modelling of the time-varying covariates processes and the event time data. This depends on the correct specification of the model for the time-varying covariates trajectories. 
  - We try to predict the trajectory of time-varying covariates in the future and extrapolate. We use that in our final model to predict $T$. 

##5.14 Other Topics (p.261)
###5.14.1 Other Survival Tree and Forest Methods

The log-rank test is weak as it can be inadequate when trying to estimate conditional survival functions (for example when two survival functions cross each other at a given time $t$ then the log-rank test won't be accurate).

The $L_1$ splitting rule can help this:

$L_1 = (n_L n_R) \int_t |\hat S_L(t) - \hat S_R (t)| dt$

where $n$ is the number of estimations in the left or right node and where $\hat S$ is the Kaplan-Meier survival function estimate.

###5.14.2 Dependent Censoring (p.263)

The analysis of survival data usually relies on the assumption that the true event time and the true censoring time are independent given the covariates. This is not always the case.

#6 Uplift Models and Estimation of Heterogeneous Treatment Effects



##6.1 Introduction to Uplift Modeling

##6.2 Two-Model And One Model With Interactions Approaches
##6.3 Class Transformation Method
##6.4 Tree-Based Methods
##6.5 Performance Criteria for Uplift Models
##6.6 An Example



# 8 Prediction Intervals

##8.1 Introduction to Prediction Intervals
##8.2 Prediction Intervals with a Linear Regression Model
##8.3 Ames Data Example (continued)
##8.4 Interpretation of a Prediction Interval

1. Marginal or heuristic sample interpretation
2. Conditional interpretation

##8.5 Post Selection Prediction Intervals
##8.6 Conformal Prediction Intervals
##8.7 Ames Data Example (continued)
##8.8 Prediction Intervals with Random Forests
##8.9 Ames Data Example (continued)







#NOT IN THE EXAM:

# 7 Models for Correlated Data

## 7.1 Customer Lifetime Value Example

## 7.2 Basic Concepts

### 7.2.1 Covariance and Correlation Matrices

### 7.2.2 Basic Setup With Clustered Data

## 7.3 Linear Mixed Models

### 7.3.1 Modeling the Covariance Structure Directly

### 7.3.2 Using Random Effects

### 7.3.3 Putting It All Together: Linear Mixed Model

### 7.3.4 Using a Linear Mixed Model For Prediction

## 7.4 Customer Lifetime Value Example (continued)

## 7.5 Generalized Linear Mixed Models

## 7.6 Variable Selection With Clustered Data

## 7.7 Customer Lifetime Value Example (continued)

## 7.8 Tree-Based Methods for Correlated Data

## 7.9 Customer Lifetime Value Example (continued)

## 7.10 Methods for Survival Correlated Data



#9 Other Topics With Random Forests

##9.1 Variable Selection with Random Forests
##9.2 Missing Data and Random Forests
###9.2.1 Handling Missing Values When Building Trees
###9.2.2 Imputation With Random Forests
##9.3 Ensemble of Rules
##9.4 Generalized Random Forests
##9.5 Transformation Forests

#10 Other Topics
##10.1 Stacking and Other Model Combination Methods
##10.2 Inference After Model Selection
##10.3 Robust Variable Selection
##10.4 Multivariate Data



























