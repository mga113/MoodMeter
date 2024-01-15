import pandas as pd
import numpy as np
import gzip, glob
import sys
from scipy import stats
import sys
import matplotlib.pyplot as plt 
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# TO RUN: python3 stat_tests.py stats_data-L

def main():
    output = sys.argv[1]
    etl_filename = glob.glob(output+'/part-*.json.gz')[0]
    etl_data = pd.read_json(etl_filename, lines=True)

    # get UBC and SFU sentiment scores 
    etl_data_UBC = etl_data[etl_data["subreddit"]=="UBC"]
    etl_data_UBC_score = etl_data_UBC["sentiment_score"]

    etl_data_SFU = etl_data[etl_data["subreddit"]=="simonfraser"]
    etl_data_SFU_score = etl_data_SFU["sentiment_score"]
    
    # Is there a difference in sentiment scores on average in UBC and SFU subreddits
    print("The p-value for OVERALL sentiment scores is:",stats.mannwhitneyu(etl_data_UBC_score,etl_data_SFU_score).pvalue)

    # Is there a difference in sentiment scores on average in POSITIVE UBC and SFU subreddits     
    # get UBC and SFU POSITIVE sentiment scores 
    etl_data_SFU_pos = etl_data_SFU[etl_data_SFU["sentiment"]=="positive"]["sentiment_score"]
    etl_data_UBC_pos = etl_data_UBC[etl_data_UBC["sentiment"]=="positive"]["sentiment_score"]
    print("The p-value for POSITIVE sentiment scores is:",stats.mannwhitneyu(etl_data_SFU_pos,etl_data_UBC_pos).pvalue)

    # Is there a difference in sentiment scores on average in NEGATIVE UBC and SFU subreddits     
    # get UBC and SFU NEGATIVE sentiment scores 
    etl_data_SFU_neg = etl_data_SFU[etl_data_SFU["sentiment"]=="negative"]["sentiment_score"]
    etl_data_UBC_neg = etl_data_UBC[etl_data_UBC["sentiment"]=="negative"]["sentiment_score"]
    print("The p-value for NEGATIVE sentiment scores is:",stats.mannwhitneyu(etl_data_SFU_neg,etl_data_UBC_neg).pvalue)

    # Is there a difference in sentiment scores on average in NEUTRAL UBC and SFU subreddits     
    # get UBC and SFU NEUTRAL sentiment scores 
    etl_data_SFU_neutral = etl_data_SFU[etl_data_SFU["sentiment"]=="neutral"]["sentiment_score"]
    etl_data_UBC_neutral = etl_data_UBC[etl_data_UBC["sentiment"]=="neutral"]["sentiment_score"]
    print("The p-value for NEUTRAL sentiment scores is:",stats.mannwhitneyu(etl_data_SFU_neutral,etl_data_UBC_neutral).pvalue)



#-----------------------------------------------------------------------------------------------------------------------------------------


    # Make a histogram to check for normality
    # If normal and with equal variances -> t-test
    # else: try to normalize

    # STEP 1) test for normality
    # Both UBC and SFU overall sentiment scores are NOT normally distributed
    print("UBC overall sentiment normality p-value: ",stats.normaltest(etl_data_UBC_score).pvalue)
    print("SFU overall sentiment normality p-value: ",stats.normaltest(etl_data_SFU_score).pvalue)

    # STEP 2) test for equal variances
    # UBC and SFU overall sentiment scores have equal variance :)
    print("UBC and SFU overall sentiment variance p-value: ",stats.levene(etl_data_UBC_score,etl_data_SFU_score).pvalue)

    # STEP 3) plot a histogram for overall UBC and SFU sentiment scores 
    plt.hist(etl_data_SFU_score, alpha=1, label='SFU')
    plt.hist(etl_data_UBC_score, alpha=0.5, label='UBC')
    plt.legend()

    # plt.show()
    plt.savefig('histogram.png')

    # STEP 4) TRANSFORM DATA TO BE NORMAL 
    '''
    The histogram illustrates bell curve disributions, so normal transformations for left/right skewed
    is not working. Despite this the normaltest shows very small p-values....non-normal results for both. 
    '''


    # STEP 5) T-TEST
    
    # The overall sentiment scores for sfu and ubc 
    ttest = stats.ttest_ind(etl_data_SFU_score,etl_data_UBC_score)
    print('T-test to determine if the overall sentiment scores for ubc/sfu are different:',ttest.pvalue)

     # The positive sentiment scores for sfu and ubc 
    ttest = stats.ttest_ind(etl_data_SFU_pos,etl_data_UBC_pos)
    print('T-test to determine if the positve sentiment scores for ubc/sfu are different:',ttest.pvalue)

     # The negative sentiment scores for sfu and ubc 
    ttest = stats.ttest_ind(etl_data_SFU_neg,etl_data_UBC_neg)
    print('T-test to determine if the negative sentiment scores for ubc/sfu are different:',ttest.pvalue)

     # The neutral sentiment scores for sfu and ubc 
    ttest = stats.ttest_ind(etl_data_SFU_neutral,etl_data_UBC_neutral)
    print('T-test to determine if the neutral sentiment scores for ubc/sfu are different:',ttest.pvalue)

    
    
    # 

    print("------------------------------------------------------------------------------------------------------------------\n")

    # create a contingency table
    contingency_table = pd.crosstab(etl_data['subreddit'], etl_data['sentiment'])

    # print the contingency table
    print("Contingency Table:")
    print(contingency_table)

    # chi-squared test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print("\nChi-squared test statistic:", chi2)
    print("P-value:", p)

    alpha = 0.05
    print("\nSignificance level (alpha):", alpha)
    print("Degrees of freedom:", dof)
    print("Expected frequencies:")
    print(expected)

    if p < alpha:
        print("\nReject the null hypothesis: There is a significant association between subreddit and sentiment.")
    else:
        print("\nFail to reject the null hypothesis: There is no significant association between subreddit and sentiment.")


    print("------------------------------------------------------------------------------------------------------------------\n")

    # extract relevant columns
    reddit_scores = etl_data['score']
    sentiment_scores = etl_data['sentiment_score']

    # scatter plot to visualize the relationship
    plt.scatter(reddit_scores, sentiment_scores)
    plt.xlabel('Reddit Scores')
    plt.ylabel('Sentiment Scores')
    plt.title('Scatter Plot of Reddit Scores vs Sentiment Scores')
    plt.savefig('correlation_scatterPlot.png')

    # calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = stats.pearsonr(reddit_scores, sentiment_scores)

    print("\nPearson Correlation Coefficient:", correlation_coefficient)
    print("P-value:", p_value)

    
    alpha = 0.05
    print("\nSignificance level (alpha):", alpha)

    if p_value < alpha:
        print("\nReject the null hypothesis: There is a significant correlation between Reddit scores and sentiment scores.")
    else:
        print("\nFail to reject the null hypothesis: There is no significant correlation between Reddit scores and sentiment scores.")


    print("------------------------------------------------------------------------------------------------------------------\n")

    data = etl_data[["month", "sentiment_score"]]

    x1 = data[data["month"] == 1]
    x2 = data[data["month"] == 2]
    x3 = data[data["month"] == 3]
    x4 = data[data["month"] == 4]
    x5 = data[data["month"] == 5]
    x6 = data[data["month"] == 6]
    x7 = data[data["month"] == 7]
    x8 = data[data["month"] == 8]
    x9 = data[data["month"] == 9]
    x10 = data[data["month"] == 10]
    x11 = data[data["month"] == 11]
    x12 = data[data["month"] == 12]

    m1 = x1["sentiment_score"].values
    m2 = x2["sentiment_score"].values
    m3 = x3["sentiment_score"].values
    m4 = x4["sentiment_score"].values
    m5 = x5["sentiment_score"].values
    m6 = x6["sentiment_score"].values
    m7 = x7["sentiment_score"].values
    m8 = x8["sentiment_score"].values
    m9 = x9["sentiment_score"].values
    m10 = x10["sentiment_score"].values
    m11 = x11["sentiment_score"].values
    m12 = x12["sentiment_score"].values

    #Run ANOVA to test if any of the months have different means

    anova = stats.f_oneway(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12)
    print("ANOVA pvalue:", anova.pvalue)

    #histogram of monthly sentiement scores
    plt.figure()
    plt.hist([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12], label=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.title("Histogram of Monthly Sentiment Scores")
    plt.savefig("monthly_sentiment_scores")
    

    #posthoc
    x1 = pd.DataFrame({"value": x1["sentiment_score"], "variable":"x1"})
    x2 = pd.DataFrame({"value": x2["sentiment_score"], "variable":"x2"})
    x3 = pd.DataFrame({"value": x3["sentiment_score"], "variable":"x3"})
    x4 = pd.DataFrame({"value": x4["sentiment_score"], "variable":"x4"})
    x5 = pd.DataFrame({"value": x5["sentiment_score"], "variable":"x5"})
    x6 = pd.DataFrame({"value": x6["sentiment_score"], "variable":"x6"})
    x7 = pd.DataFrame({"value": x7["sentiment_score"], "variable":"x7"})
    x8 = pd.DataFrame({"value": x8["sentiment_score"], "variable":"x8"})
    x9 = pd.DataFrame({"value": x9["sentiment_score"], "variable":"x9"})
    x10 = pd.DataFrame({"value": x10["sentiment_score"], "variable":"x10"})
    x11 = pd.DataFrame({"value": x11["sentiment_score"], "variable":"x11"})
    x12 = pd.DataFrame({"value": x12["sentiment_score"], "variable":"x12"})

    months = pd.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], ignore_index=True) 

    posthoc = pairwise_tukeyhsd(
    months['value'], months['variable'],
    alpha=0.05)

    fig = posthoc.plot_simultaneous()
    plt.savefig("posthoc")
    print(posthoc)

    #Plotting monthly sentiment score averages graph
    d = etl_data[['month', 'sentiment_score']]
    avg_by_month = d.groupby('month')['sentiment_score'].mean()

    plt.figure()
    avg_by_month.plot(marker='o', linestyle='-')
    plt.title('Average Sentiment Score by Month in 2016')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.savefig("Monthly Avg Scores")



main()

