## **Final Project Submission**

Please fill out:
* Student name: Darius Fuller
* Student pace: Part-time
* Scheduled project review date/time: TBD 
* Instructor name: James Irving
* Blog post URL:https://medium.com/@d_full22/working-with-northwind-66a689a3b0fe

## Setting Up


```python
import scipy.stats as stats
import statsmodels.api as sms
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import functions as fn
import sqlite3
import warnings
warnings.filterwarnings('ignore')
```


```python
conn = sqlite3.Connection('Northwind_small.sqlite')
cur = sqlite3.Cursor(conn)
```

![png](Images/Northwind_ERD_updated.png)


```python
cur.execute('''SELECT name 
               FROM sqlite_master
               WHERE type='table';''')

df_table = pd.DataFrame(cur.fetchall(), columns=['Table'])

df_table # Verifying the images results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Employee</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Category</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Customer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Shipper</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Supplier</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Order</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Product</td>
    </tr>
    <tr>
      <th>7</th>
      <td>OrderDetail</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CustomerCustomerDemo</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CustomerDemographic</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Region</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Territory</td>
    </tr>
    <tr>
      <th>12</th>
      <td>EmployeeTerritory</td>
    </tr>
  </tbody>
</table>
</div>



## Q1: **Does discount amount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount?**

* $H_0$: Discount amount does not significantly effect the amount of products ordered.
* $H_1$: Discounts do have a significant effect (positive or negative) on the amount of products ordered.

> For this test, I will be using a two-tailed test.

### Initial Query & Feature Engineering


```python
cur.execute('''SELECT *
               FROM OrderDetail o;''')
df1 = pd.DataFrame(cur.fetchall()) 
df1.columns = [i[0] for i in cur.description]
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



> Here I look to create a boolean column to indicated if a column has been discounted or not.


```python
df1['Discounted'] = df1['Discount'] > 0
```


```python
df1['Discounted'].sum() # How many discounted (True) orders are there? 
```




    838




```python
df1.head() # Quick check
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Discounted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### EDA

> Checking to see whether or not there is a visual difference between the two groups: Discounted (True) or Full Price (False)


```python
fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(data=df1, x='Discounted', y='Quantity', ci=68, ax=ax)
sns.set(style='darkgrid')
plt.title('Avg. Quantity: Discounted vs Full Price');
```


![png](Images/Mod3_Proj_17_0.png)


> In order to have a more flexible instance of my target data, I need to move the information from the DataFrame into a python dictionary.


```python
quantVSdisc = {}
for item in df1['Discounted'].unique():
    quantVSdisc[item] = df1.groupby('Discounted').get_group(item)['Quantity']
```


```python
print(len(quantVSdisc[False]), len(quantVSdisc[True])) 
# Confirming results transferred to dictionary
```

    1317 838
    

> It is clear, *visually*, that there may be a valid effect on the quantity purchased if the order is discounted. There are two groups of interest and thus I will be checking the assumptions for a **2 Sample T-test**.

### Testing Assumptions (2 Sample T-test)

#### Outliers?


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in quantVSdisc.items():
    if key == True:
        lab = 'Discounted'
    else:
        lab = 'Full Price'
        
    sns.distplot(val, label=key, ax=ax)

plt.title('Quantity: Discounted vs Full Price')
ax.legend();

# Visual check for skew
```


![png](Images/Mod3_Proj_24_0.png)


> Although skewed, the tails do not look very thick, will remove some outliers to try for better normality since I have a lot of data still.


```python
for key, val in quantVSdisc.items():
    out_dict = fn.find_outliers_Z(val)
    print(f'There are {out_dict.sum()} {key} Z-outliers.')
    
    out_dict = fn.find_outliers_IQR(val)
    print(f'There are {out_dict.sum()} {key} IQR-outliers.')
```

    There are 20 False Z-outliers.
    There are 34 False IQR-outliers.
    There are 15 True Z-outliers.
    There are 29 True IQR-outliers.
    

> Checking IQR method for posterity, although I intend to use the Z-score method to remain conservative with removal of data.


```python
for key, val in quantVSdisc.items():
    out_dict = fn.find_outliers_Z(val)
    quantVSdisc[key] = val[~out_dict]
    
# Removing values outside of +/- 3 Z-scores from mean directly from dictionary
```


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in quantVSdisc.items():
    if key == True:
        lab = 'Discounted'
    else:
        lab = 'Full Price'
        
    sns.distplot(val, label=lab, ax=ax)

plt.title('Quantity: Discounted vs Full Price sans Outliers')
ax.legend();

# 2nd visual check (now without outliers)
```


![png](Images/Mod3_Proj_29_0.png)


> The data looks a lot closer to standard normal. Now I can move forward to test normality.

#### Normality?


```python
for key, val in quantVSdisc.items():
    stat, p = stats.normaltest(val)
    
    if key == True:
        lab = 'Discounted'
    else:
        lab = 'Full Price'
    
    print(f'{lab} normal test p-value = {round(p,4)}')
    
    sig = 'is NOT' if p < .05 else 'IS'
    
    print(f'The data {sig} normal.')
```

    Full Price normal test p-value = 0.0
    The data is NOT normal.
    Discounted normal test p-value = 0.0
    The data is NOT normal.
    


```python
print(len(quantVSdisc[False]), len(quantVSdisc[True]))
```

    1297 823
    

> The data is *not* normal in either sample. This means to move forward I need to have samples larger than 15 each (recommended). Since I have 1,297 Full Price and 823 Discounted I can do so.

#### Equal Variance?

> In order for stats.levene() to properly accept my data, I need to unpack my dictionary into a list.


```python
norm_list = []

for key, val in quantVSdisc.items():
    norm_list.append(val)
```


```python
stat, p = stats.levene(*norm_list)

print(f'Levene test p-value = {round(p,4)}')

sig = 'does NOT' if p < .05 else 'DOES'

print(f'The data {sig} have equal variance.')
```

    Levene test p-value = 0.0
    The data does NOT have equal variance.
    

> Since the Levene test was failed, I must use the Welch's T-test function with the 'equal_var' parameter set to 'False' in order to determine whether or not I am dealing with two samples from different populations and test my hypothesis.

### Hypothesis Test


```python
stat, p = stats.ttest_ind(*norm_list, equal_var=False)

print(f"Welch's T-test p-value = {round(p,4)}")

sig = 'IS' if p < .05 else 'is NOT'

print(f'The data {sig} from different populations.')
```

    Welch's T-test p-value = 0.0
    The data IS from different populations.
    

> The data does *NOT* have equal variance but *IS* from different populations. Given the information above, I can move forward with **Rejecting the Null Hypothesis ($H_0$)**.

### Post-Hoc Calculations


```python
eff_size = fn.Cohen_d(quantVSdisc[True], quantVSdisc[False])

eff_size
```




    0.32001140965727837



> Given the standard interpretation of Cohen's D, our value falls between the 'small effect' category (0.2) and 'medium effect' category (0.5). Therefore we can say, although relatively small, discounts **do** have an effect on the quantity purchased.


```python
model = pairwise_tukeyhsd(df1['Quantity'], df1['Discount'])
model.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD,FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>   <th>lower</th>   <th>upper</th>  <th>reject</th>
</tr>
<tr>
    <td>0.0</td>   <td>0.01</td>  <td>-19.7153</td> <td>-80.3306</td> <td>40.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.02</td>  <td>-19.7153</td>  <td>-62.593</td> <td>23.1625</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.03</td>  <td>-20.0486</td> <td>-55.0714</td> <td>14.9742</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.04</td>  <td>-20.7153</td> <td>-81.3306</td> <td>39.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.05</td>   <td>6.2955</td>   <td>1.5381</td>  <td>11.053</td>   <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.06</td>  <td>-19.7153</td> <td>-80.3306</td> <td>40.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.1</td>   <td>3.5217</td>   <td>-1.3783</td> <td>8.4217</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.15</td>   <td>6.6669</td>    <td>1.551</td>  <td>11.7828</td>  <td>True</td> 
</tr>
<tr>
    <td>0.0</td>    <td>0.2</td>   <td>5.3096</td>   <td>0.2508</td>  <td>10.3684</td>  <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.25</td>    <td>6.525</td>   <td>1.3647</td>  <td>11.6852</td>  <td>True</td> 
</tr>
<tr>
   <td>0.01</td>   <td>0.02</td>     <td>0.0</td>   <td>-74.2101</td> <td>74.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.03</td>   <td>-0.3333</td> <td>-70.2993</td> <td>69.6326</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.04</td>    <td>-1.0</td>   <td>-86.6905</td> <td>84.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.05</td>   <td>26.0108</td>  <td>-34.745</td> <td>86.7667</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.06</td>     <td>0.0</td>   <td>-85.6905</td> <td>85.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.1</td>   <td>23.237</td>  <td>-37.5302</td> <td>84.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.15</td>   <td>26.3822</td> <td>-34.4028</td> <td>87.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.2</td>   <td>25.0248</td> <td>-35.7554</td> <td>85.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.25</td>   <td>26.2403</td> <td>-34.5485</td> <td>87.029</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.03</td>   <td>-0.3333</td> <td>-55.6463</td> <td>54.9796</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.04</td>    <td>-1.0</td>   <td>-75.2101</td> <td>73.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.05</td>   <td>26.0108</td> <td>-17.0654</td> <td>69.087</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.06</td>     <td>0.0</td>   <td>-74.2101</td> <td>74.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.1</td>   <td>23.237</td>  <td>-19.8552</td> <td>66.3292</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.15</td>   <td>26.3822</td> <td>-16.7351</td> <td>69.4994</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.2</td>   <td>25.0248</td> <td>-18.0857</td> <td>68.1354</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.25</td>   <td>26.2403</td> <td>-16.8823</td> <td>69.3628</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.04</td>   <td>-0.6667</td> <td>-70.6326</td> <td>69.2993</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.05</td>   <td>26.3441</td>  <td>-8.9214</td> <td>61.6096</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.06</td>   <td>0.3333</td>  <td>-69.6326</td> <td>70.2993</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.1</td>   <td>23.5703</td> <td>-11.7147</td> <td>58.8553</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.15</td>   <td>26.7155</td>  <td>-8.6001</td> <td>62.0311</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.2</td>   <td>25.3582</td>  <td>-9.9492</td> <td>60.6656</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.25</td>   <td>26.5736</td>  <td>-8.7485</td> <td>61.8957</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.05</td>   <td>27.0108</td>  <td>-33.745</td> <td>87.7667</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.06</td>     <td>1.0</td>   <td>-84.6905</td> <td>86.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.1</td>   <td>24.237</td>  <td>-36.5302</td> <td>85.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.15</td>   <td>27.3822</td> <td>-33.4028</td> <td>88.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.2</td>   <td>26.0248</td> <td>-34.7554</td> <td>86.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.25</td>   <td>27.2403</td> <td>-33.5485</td> <td>88.029</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.06</td>  <td>-26.0108</td> <td>-86.7667</td> <td>34.745</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.1</td>   <td>-2.7738</td>  <td>-9.1822</td> <td>3.6346</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.15</td>   <td>0.3714</td>   <td>-6.2036</td> <td>6.9463</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.2</td>   <td>-0.986</td>   <td>-7.5166</td> <td>5.5447</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.25</td>   <td>0.2294</td>   <td>-6.3801</td>  <td>6.839</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.1</td>   <td>23.237</td>  <td>-37.5302</td> <td>84.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.15</td>   <td>26.3822</td> <td>-34.4028</td> <td>87.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.2</td>   <td>25.0248</td> <td>-35.7554</td> <td>85.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.25</td>   <td>26.2403</td> <td>-34.5485</td> <td>87.029</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.15</td>   <td>3.1452</td>   <td>-3.5337</td>  <td>9.824</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>    <td>0.2</td>   <td>1.7879</td>   <td>-4.8474</td> <td>8.4231</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.25</td>   <td>3.0033</td>   <td>-3.7096</td> <td>9.7161</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>    <td>0.2</td>   <td>-1.3573</td>  <td>-8.1536</td> <td>5.4389</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>   <td>0.25</td>   <td>-0.1419</td>  <td>-7.014</td>  <td>6.7302</td>   <td>False</td>
</tr>
<tr>
    <td>0.2</td>   <td>0.25</td>   <td>1.2154</td>   <td>-5.6143</td> <td>8.0451</td>   <td>False</td>
</tr>
</table>



> Given an alpha level of 0.05, I can **reject** the null hypothesis for the following discount levels:
* 5%
* 15%
* 20%
* 25%

### Q1 Summary

> After testing my hypothesis regarding quantity on discounted orders, I can say to stakeholders that discounting does in fact have, *with 95% confidence*, an effect on the quantity of any given order. 

> Additionally, I can say that the discount levels (in descending order) of 15%, 25%, 5%, 20% have the most significant effect on the avg. quantity of a given order. 

## Q2: Does discount amount have a statistically significant effect on the total amount spent in an order? If so, at what level(s) of discount?

* $H_0$: Discount amount does not significantly effect the total amount of money spent.
* $H_1$: Discounts do have a significant effect (positive or negative) on the total amount of money spent.

> For this test, I will be using a two-tailed test.

### Initial Query & Feature Engineering

> I will do a very similar setup to the previous hypothesis for the quantity exploration.


```python
cur.execute('''SELECT *
               FROM OrderDetail o;''')
df2 = pd.DataFrame(cur.fetchall()) 
df2.columns = [i[0] for i in cur.description]
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2['Discounted'] = df2['Discount'] > 0
# Creating boolean column for discount or full price
```


```python
df2['TotalSpent'] = (df2['UnitPrice'] * (1-df2['Discount'])) * df2['Quantity']
# Creating a numerical column for total price spent per order (including discount)
```


```python
df2.head() # Quick check
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Discounted</th>
      <th>TotalSpent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>False</td>
      <td>168.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>False</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>False</td>
      <td>174.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>False</td>
      <td>167.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>False</td>
      <td>1696.0</td>
    </tr>
  </tbody>
</table>
</div>



### EDA


```python
# Visual check

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(data=df2, x='Discounted', y='TotalSpent', ci=68, ax=ax)
sns.set(style='darkgrid')
plt.title('Avg. TotalSpent: Discounted vs Full Price');
```


![png](Images/Mod3_Proj_60_0.png)


> Converting to a dictionary for flexibility (as before)


```python
totspentVSdisc = {}
for item in df2['Discounted'].unique():
    totspentVSdisc[item] = df2.groupby('Discounted').get_group(item)['TotalSpent']
```


```python
print(len(totspentVSdisc[False]), len(totspentVSdisc[True]))
# Verifying results
```

    1317 838
    

> There is, once again *visually*, a small difference in the average total spent per order. Not too hopeful of its significance, but it is still worth investigation. As with my first hypothesis, this will fall under a **2 Sample T-test**.

### Testing Assumptions (2 Sample T-test)

#### Outliers?


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in totspentVSdisc.items():
    if key == True:
        lab = 'Discounted'
    else:
        lab = 'Full Price'
        
    sns.distplot(val, label=key, ax=ax)

plt.title('Total Spent: Discounted vs Full Price')
ax.legend();

# Visual check for skew
```


![png](Images/Mod3_Proj_67_0.png)


> This is *highly* skewed to the right, although very thin. 


```python
for key, val in totspentVSdisc.items():
    out_dict = fn.find_outliers_Z(val)
    print(f'There are {out_dict.sum()} {key} Z-outliers.')
    
    out_dict = fn.find_outliers_IQR(val)
    print(f'There are {out_dict.sum()} {key} IQR-outliers.')
```

    There are 19 False Z-outliers.
    There are 101 False IQR-outliers.
    There are 16 True Z-outliers.
    There are 66 True IQR-outliers.
    

> Despite there being a very long tail, I will for consistency stick with Z-scores in order to remain conservative.


```python
for key, val in totspentVSdisc.items():
    out_dict = fn.find_outliers_Z(val)
    totspentVSdisc[key] = val[~out_dict]
    
# Removing values outside of +/- 3 Z-scores from mean directly from dictionary
```


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in totspentVSdisc.items():
    if key == True:
        lab = 'Discounted'
    else:
        lab = 'Full Price'
        
    sns.distplot(val, label=lab, ax=ax)

plt.title('Total Spent: Discounted vs Full Price sans Outliers')
ax.legend();

# 2nd visual check (now without outliers)
```


![png](Images/Mod3_Proj_72_0.png)


> Although, if compared to the standard normal distribution, the data is still pretty skewed to the right. I will move forward with testing assumptions.

#### Normality?


```python
for key, val in totspentVSdisc.items():
    stat, p = stats.normaltest(val)
    
    if key == True:
        lab = 'Discounted'
    else:
        lab = 'Full Price'
    
    print(f'{lab} normal test p-value = {round(p,4)}')
    
    sig = 'is NOT' if p < .05 else 'IS'
    
    print(f'The data {sig} normal.')
```

    Full Price normal test p-value = 0.0
    The data is NOT normal.
    Discounted normal test p-value = 0.0
    The data is NOT normal.
    

> No surprises here, this data is *not* normal.


```python
print(len(totspentVSdisc[False]), len(totspentVSdisc[True]))
```

    1298 822
    

> Luckily, with Full Price group having 1,298 and Discounted having 822, I am able to ignore the need for normality and move on to check for equal variance. 

#### Equal Variance?


```python
# Same as before

norm_list2 = []

for key, val in totspentVSdisc.items():
    norm_list2.append(val)
```


```python
stat, p = stats.levene(*norm_list2)

print(f'Levene test p-value = {round(p,4)}')

sig = 'does NOT' if p < .05 else 'DOES'

print(f'The data {sig} have equal variance.')
```

    Levene test p-value = 0.4614
    The data DOES have equal variance.
    

> Since the data does pass all of the assumption tests, I can move forward with a normal 2 Sample T-test.

### Hypothesis Test


```python
stat, p = stats.ttest_ind(*norm_list2)

print(f"Welch's T-test p-value = {round(p,4)}")

sig = 'IS' if p < .05 else 'is NOT'

print(f'The data {sig} from different populations.')
```

    Welch's T-test p-value = 0.2057
    The data is NOT from different populations.
    

> At this point, I have **failed to reject** the null hypothesis.

### Post-Hoc Calculations

> Although I have failed to reject the null, I want to investigate if at a given level, a discount *does* have a significant effect upon the total amount spent.


```python
model2 = pairwise_tukeyhsd(df2['TotalSpent'], df2['Discount'])
model2.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD,FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>     <th>lower</th>     <th>upper</th>   <th>reject</th>
</tr>
<tr>
    <td>0.0</td>   <td>0.01</td>  <td>-540.3065</td> <td>-3662.2549</td> <td>2581.6418</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.02</td>  <td>-540.1165</td> <td>-2748.5047</td> <td>1668.2716</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.03</td>  <td>-529.703</td>  <td>-2333.5278</td> <td>1274.1217</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.04</td>  <td>-492.2465</td> <td>-3614.1949</td> <td>2629.7018</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.05</td>  <td>227.9252</td>   <td>-17.1036</td>   <td>472.954</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.06</td>  <td>-506.0865</td> <td>-3628.0349</td> <td>2615.8618</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.1</td>  <td>-41.1098</td>    <td>-293.48</td>  <td>211.2604</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.15</td>  <td>-12.6424</td>   <td>-276.1341</td> <td>250.8493</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.2</td>  <td>-16.0866</td>   <td>-276.6374</td> <td>244.4641</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.25</td>   <td>72.4517</td>   <td>-193.3232</td> <td>338.2266</td>   <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.02</td>    <td>0.19</td>    <td>-3821.9494</td> <td>3822.3294</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.03</td>   <td>10.6035</td>  <td>-3592.9441</td> <td>3614.1511</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.04</td>    <td>48.06</td>   <td>-4365.3664</td> <td>4461.4864</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.05</td>  <td>768.2318</td>  <td>-2360.9551</td> <td>3897.4187</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.06</td>    <td>34.22</td>   <td>-4379.2064</td> <td>4447.6464</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.1</td>  <td>499.1968</td>  <td>-2630.5736</td> <td>3628.9671</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.15</td>  <td>527.6642</td>  <td>-2603.0226</td> <td>3658.3509</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.2</td>  <td>524.2199</td>  <td>-2606.2207</td> <td>3654.6605</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.25</td>  <td>612.7582</td>  <td>-2518.1215</td> <td>3743.638</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.03</td>   <td>10.4135</td>   <td>-2838.441</td> <td>2859.268</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.04</td>    <td>47.87</td>   <td>-3774.2694</td> <td>3870.0094</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.05</td>  <td>768.0418</td>  <td>-1450.5676</td> <td>2986.6511</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.06</td>    <td>34.03</td>   <td>-3788.1094</td> <td>3856.1694</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.1</td>  <td>499.0068</td>  <td>-1720.4254</td> <td>2718.4389</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.15</td>  <td>527.4742</td>  <td>-1693.2501</td> <td>2748.1984</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.2</td>  <td>524.0299</td>  <td>-1696.3473</td> <td>2744.4071</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.25</td>  <td>612.5682</td>  <td>-1608.4281</td> <td>2833.5645</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.04</td>   <td>37.4565</td>  <td>-3566.0911</td> <td>3641.0041</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.05</td>  <td>757.6283</td>  <td>-1058.6958</td> <td>2573.9523</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.06</td>   <td>23.6165</td>  <td>-3579.9311</td> <td>3627.1641</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.1</td>  <td>488.5933</td>  <td>-1328.7357</td> <td>2305.9222</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.15</td>  <td>517.0607</td>  <td>-1301.8461</td> <td>2335.9674</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.2</td>  <td>513.6164</td>  <td>-1304.8666</td> <td>2332.0994</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.25</td>  <td>602.1547</td>  <td>-1217.0842</td> <td>2421.3936</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.05</td>  <td>720.1718</td>  <td>-2409.0151</td> <td>3849.3587</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.06</td>   <td>-13.84</td>   <td>-4427.2664</td> <td>4399.5864</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.1</td>  <td>451.1368</td>  <td>-2678.6336</td> <td>3580.9071</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.15</td>  <td>479.6042</td>  <td>-2651.0826</td> <td>3610.2909</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.2</td>  <td>476.1599</td>  <td>-2654.2807</td> <td>3606.6005</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.25</td>  <td>564.6982</td>  <td>-2566.1815</td> <td>3695.578</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.06</td>  <td>-734.0118</td> <td>-3863.1987</td> <td>2395.1751</td>  <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.1</td>  <td>-269.035</td>   <td>-599.0955</td>  <td>61.0255</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.15</td>  <td>-240.5676</td>  <td>-579.2076</td>  <td>98.0724</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.2</td>  <td>-244.0119</td>  <td>-580.3686</td>  <td>92.3449</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.25</td>  <td>-155.4735</td>  <td>-495.8931</td>  <td>184.946</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.1</td>  <td>464.9768</td>  <td>-2664.7936</td> <td>3594.7471</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.15</td>  <td>493.4442</td>  <td>-2637.2426</td> <td>3624.1309</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.2</td>  <td>489.9999</td>  <td>-2640.4407</td> <td>3620.4405</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.25</td>  <td>578.5382</td>  <td>-2552.3415</td> <td>3709.418</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.15</td>   <td>28.4674</td>   <td>-315.5219</td> <td>372.4568</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>    <td>0.2</td>   <td>25.0231</td>   <td>-316.7187</td>  <td>366.765</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.25</td>  <td>113.5615</td>   <td>-232.1799</td> <td>459.3029</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>    <td>0.2</td>   <td>-3.4443</td>   <td>-353.4794</td> <td>346.5909</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>   <td>0.25</td>   <td>85.0941</td>   <td>-268.847</td>  <td>439.0351</td>   <td>False</td>
</tr>
<tr>
    <td>0.2</td>   <td>0.25</td>   <td>88.5383</td>   <td>-263.2188</td> <td>440.2954</td>   <td>False</td>
</tr>
</table>



### Q2 Summary

> After testing my hypothesis regarding the total amount spent on discounted orders, I can say to stakeholders that discounting does **not** have, *with 95% confidence*, an effect on the total amount spent on any given order. 

> Additionally, I can say that no matter the discount level, there is no significant effect on the avg. amount spent on a given order. 

## Q3: Does the region in which the products are sold have a significant effect on the quantity of a product in an order? If so, which region buys the most?

* $H_0$: RegionID does not significantly effect the amount of products ordered.
* $H_1$: RegionID does have a significant effect (positive or negative) on the amount of products ordered.

> For this test, I will be using a two-tailed test.

### Initial Query & Feature Engineering


```python
cur.execute('''SELECT DISTINCT(o.OrderId), o.Quantity, r.Id, r.RegionDescription
               FROM OrderDetail o
               JOIN 'Order' ord
               ON o.OrderId = ord.Id
               JOIN Employee e
               ON ord.EmployeeId = e.Id
               JOIN EmployeeTerritory et
               USING(EmployeeId)
               JOIN Territory t
               ON et.TerritoryId = t.Id
               JOIN Region r
               ON r.Id = t.RegionId;''') 
df3 = pd.DataFrame(cur.fetchall()) 
df3.columns = [i[0] for i in cur.description]
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OrderId</th>
      <th>Quantity</th>
      <th>Id</th>
      <th>RegionDescription</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248</td>
      <td>12</td>
      <td>1</td>
      <td>Eastern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10248</td>
      <td>10</td>
      <td>1</td>
      <td>Eastern</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10248</td>
      <td>5</td>
      <td>1</td>
      <td>Eastern</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10249</td>
      <td>9</td>
      <td>2</td>
      <td>Western</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10249</td>
      <td>40</td>
      <td>2</td>
      <td>Western</td>
    </tr>
  </tbody>
</table>
</div>



> I ran into a duplication error due to (likely) the Order Id and Employee Id's being one-to-many during the joining process; coercing the OrderIds to be *distinct* should solve this.

### EDA

> **Region Descriptors**
1. 'Eastern'
2. 'Western'
3. 'Northern'
4. 'Southern'


```python
fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(data=df3, x='RegionDescription', y='Quantity', ci=68, ax=ax)

plt.title('Avg. Quantity: By Region');
```


![png](Images/Mod3_Proj_99_0.png)



```python
# Quick check of differences via swarmplot

#fig, ax = plt.subplots(figsize=(8,5))

sns.catplot(data=df3, x='RegionDescription', y='Quantity', kind='swarm', ci=68)

plt.title('Quantity: By Region');
```


![png](Images/Mod3_Proj_100_0.png)



```python
quantVSreg = {}
for item in df3['RegionDescription'].unique():
    quantVSreg[item] = df3.groupby('RegionDescription').get_group(item)['Quantity']
```


```python
for item in df3['RegionDescription'].unique():
    print(f'Number of instances in {item}: {len(quantVSreg[item])}')
```

    Number of instances in Eastern: 1022
    Number of instances in Western: 325
    Number of instances in Southern: 302
    Number of instances in Northern: 349
    

> Visually there is not a large difference between regions, however my goal is simply to find out whether or not an effect exists. It is clear that the Eastern region does make up almost 50% of the orders they received during the period of time this data represents. Due to our data being more than two groups, I will need to proceed with a **One-Way ANOVA** test.

### Testing Assumptions (One-Way ANOVA)

#### Outliers?


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in quantVSreg.items():       
    sns.distplot(val, label=key, ax=ax)

plt.title('Quantity: by Region')
ax.legend();

# Visual check for skew
```


![png](Images/Mod3_Proj_106_0.png)


> All of the groups appear to have a similar level of skew and weight in the right tail. There is one item for the Eastern group that hangs pretty far into the tail, I am interested to see whether or not it will fall off after outliers are removed.


```python
for key, val in quantVSreg.items():
    out_dict = fn.find_outliers_Z(val)
    print(f'There are {out_dict.sum()} {key} Z-outliers.')
    
    #out_dict = fn.find_outliers_IQR(val)
    #print(f'There are {out_dict.sum()} {key} IQR-outliers.')
```

    There are 14 Eastern Z-outliers.
    There are 5 Western Z-outliers.
    There are 4 Southern Z-outliers.
    There are 8 Northern Z-outliers.
    


```python
# Removing outliers per the Z-score method

for key, val in quantVSreg.items():
    out_dict = fn.find_outliers_Z(val)
    quantVSreg[key] = val[~out_dict]
```


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in quantVSreg.items():
    sns.distplot(val, label=key, ax=ax)

plt.title('Quantity: by Region sans Outliers')
ax.legend();

# 2nd visual check (now without outliers)
```


![png](Images/Mod3_Proj_110_0.png)


> My function removed a large bit of the tails, and visually I can see that the Northern/Eastern groups do not skew out as far as the other two groups.

#### Normality?


```python
for key, val in quantVSreg.items():
    stat, p = stats.normaltest(val)
        
    print(f'{key} normal test p-value = {round(p,4)}')
    
    sig = 'is NOT' if p < .05 else 'IS'
    
    print(f'The data {sig} normal.')
```

    Eastern normal test p-value = 0.0
    The data is NOT normal.
    Western normal test p-value = 0.0
    The data is NOT normal.
    Southern normal test p-value = 0.0
    The data is NOT normal.
    Northern normal test p-value = 0.0
    The data is NOT normal.
    


```python
for item in df3['RegionDescription'].unique():
    print(f'Number of instances in {item}: {len(quantVSreg[item])}')
```

    Number of instances in Eastern: 1008
    Number of instances in Western: 320
    Number of instances in Southern: 298
    Number of instances in Northern: 341
    

> Not a single group is considered normal. However, due to the groups all having more than 15 instances (by far), I am able to ignore this assumption and test for equal variance.

#### Equal Variance?


```python
norm_list3 = []

for key, val in quantVSreg.items():
    norm_list3.append(val)
```


```python
stat, p = stats.levene(*norm_list3)

print(f'Levene test p-value = {round(p,4)}')

sig = 'does NOT' if p < .05 else 'DOES'

print(f'The data {sig} have equal variance.')
```

    Levene test p-value = 0.11
    The data DOES have equal variance.
    

> Since the data passed Levene's test, I am able to safely move forward with a standard One-Way ANOVA in order to see if my samples are from different populations and test my hypothesis.

### Hypothesis Test


```python
stat, p = stats.f_oneway(*norm_list3)

print(f"One-Way ANOVA p-value = {round(p,4)}")

sig = 'IS' if p < .05 else 'is NOT'

print(f'The data {sig} from different populations.')
```

    One-Way ANOVA p-value = 0.3789
    The data is NOT from different populations.
    

> The data does *NOT* have a p-value low enought to say confidently that it is from different populations. Given the information above, I have **failed to reject the Null Hypothesis ($H_0$)**.

### Post-Hoc Calculations

> Although I have failed to reject the null, (as with Q2) I want to investigate if inter-regionally, there *is* have a significant effect upon the average quantity ordered.


```python
model3 = pairwise_tukeyhsd(df3['Quantity'], df3['RegionDescription'])
model3.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD,FWER=0.05</caption>
<tr>
   <th>group1</th>   <th>group2</th>  <th>meandiff</th>  <th>lower</th>   <th>upper</th> <th>reject</th>
</tr>
<tr>
   <td>Eastern</td> <td>Northern</td>  <td>-1.0508</td> <td>-4.1218</td> <td>2.0202</td>  <td>False</td>
</tr>
<tr>
   <td>Eastern</td> <td>Southern</td>  <td>0.2852</td>  <td>-2.959</td>  <td>3.5295</td>  <td>False</td>
</tr>
<tr>
   <td>Eastern</td>  <td>Western</td>  <td>-0.8925</td> <td>-4.0468</td> <td>2.2619</td>  <td>False</td>
</tr>
<tr>
  <td>Northern</td> <td>Southern</td>   <td>1.336</td>  <td>-2.5569</td> <td>5.2289</td>  <td>False</td>
</tr>
<tr>
  <td>Northern</td>  <td>Western</td>  <td>0.1583</td>   <td>-3.66</td>  <td>3.9766</td>  <td>False</td>
</tr>
<tr>
  <td>Southern</td>  <td>Western</td>  <td>-1.1777</td> <td>-5.1367</td> <td>2.7813</td>  <td>False</td>
</tr>
</table>



### Q3 Summary

> After testing my hypothesis regarding the average quantity ordered per region, I can say to stakeholders that there is **not**, *with 95% confidence*, an effect on the average quantity purchased in a given order. 

> Additionally, I can say that between individual regions, there is no evidence to suggest a significant difference in the average quantity purchased in a given order. 

## Q4: Does whether the company keeps an item 'in stock' or not have any effect on the total spent on those products? If so, is there an optimal level to reorder?

* $H_0$: The level at which Northwind re-orders products has no effect on the total amount spent on orders including those items.
* $H_1$: Whether or not Northwind waits until they are out of stock to reorder has an effect on the total amount spent on orders with those items.

> For this test, I will be using a two-tailed test.

### Initial Query & Feature Engineering


```python
cur.execute('''SELECT ord.ProductId, ord.Quantity, ord.UnitPrice, 
               ord.Discount, p.ReorderLevel, p.CategoryId, 
               ((ord.Quantity * ord.UnitPrice)*(1 - ord.Discount)) 
               AS TotalSpent
               FROM OrderDetail ord
               JOIN Product p
               ON ord.ProductId = p.Id;''')
df4 = pd.DataFrame(cur.fetchall()) 
df4.columns = [i[0] for i in cur.description]
df4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ProductId</th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>Discount</th>
      <th>ReorderLevel</th>
      <th>CategoryId</th>
      <th>TotalSpent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>12</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>30</td>
      <td>4</td>
      <td>168.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>10</td>
      <td>9.8</td>
      <td>0.0</td>
      <td>0</td>
      <td>5</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>5</td>
      <td>34.8</td>
      <td>0.0</td>
      <td>0</td>
      <td>4</td>
      <td>174.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>9</td>
      <td>18.6</td>
      <td>0.0</td>
      <td>0</td>
      <td>7</td>
      <td>167.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>40</td>
      <td>42.4</td>
      <td>0.0</td>
      <td>10</td>
      <td>7</td>
      <td>1696.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4.ReorderLevel.value_counts() # Quick check of samples 
```




    0     718
    15    273
    25    256
    10    242
    30    235
    20    235
    5     196
    Name: ReorderLevel, dtype: int64




```python
df4.CategoryId.value_counts() # Spread check among categories (for later)
```




    1    404
    4    366
    3    334
    8    330
    2    216
    5    196
    6    173
    7    136
    Name: CategoryId, dtype: int64



### EDA

> **Reorder Levels**
* 0 - 30 at five unit intervals (0, 5, 10,...30) 


```python
fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(data=df4, x='ReorderLevel', y='TotalSpent', ci=68, ax=ax)

plt.title('Avg. TotalSpent: By ReorderLevel');
```


![png](Images/Mod3_Proj_137_0.png)



```python
# Quick check of differences via swarmplot (like above)

#fig, ax = plt.subplots(figsize=(8,5))

sns.catplot(data=df4, x='ReorderLevel', y='TotalSpent', kind='swarm', ci=68)

plt.title('TotalSpent: By ReorderLevel');
```


![png](Images/Mod3_Proj_138_0.png)



```python
totspentVSreord = {}
for item in df4['ReorderLevel'].unique():
    totspentVSreord[item] = df4.groupby('ReorderLevel').get_group(item)['TotalSpent']
```


```python
# Quick check for correct data transfer

for item in sorted(df4['ReorderLevel'].unique()):
    print(f'Number of instances in {item}: {len(totspentVSreord[item])}')
    
# We're good!
```

    Number of instances in 0: 718
    Number of instances in 5: 196
    Number of instances in 10: 242
    Number of instances in 15: 273
    Number of instances in 20: 235
    Number of instances in 25: 256
    Number of instances in 30: 235
    

> Visually, I can see that those at a 0 and 15 reorder level tend to stick out with the bar chart. The swarm chart demonstrates why the error bar is so large on the bar chart, the 15 unit reorder level seems to have a wide range of values. I will need to use the **One-Way ANOVA** test since I will be comparing multiple sample populations.

### Testing Assumptions (One-Way ANOVA)

#### Outliers?


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in sorted(totspentVSreord.items()):       
    sns.distplot(val, label=key, ax=ax)

plt.title('TotalSpent: by Reorder Level')
ax.legend();

# Visual check for skew
```


![png](Images/Mod3_Proj_144_0.png)


> The extreme values in one or more of the reorder levels are skewing the distribution strongly. It seems best to remove them to ensure I get closer in the normality test.


```python
for key, val in sorted(totspentVSreord.items()):
    out_dict = fn.find_outliers_Z(val)
    print(f'There are {out_dict.sum()} {key} Z-outliers.')
    
    #out_dict = fn.find_outliers_IQR(val)
    #print(f'There are {out_dict.sum()} {key} IQR-outliers.')
    
    #due to the number of categories, no need to check IQR
```

    There are 15 0 Z-outliers.
    There are 5 5 Z-outliers.
    There are 3 10 Z-outliers.
    There are 9 15 Z-outliers.
    There are 5 20 Z-outliers.
    There are 5 25 Z-outliers.
    There are 3 30 Z-outliers.
    

> Surprisingly, the 0 level has the most values outside of the +/- 3 z-score threshold.


```python
# Removing outliers per the Z-score method

for key, val in totspentVSreord.items():
    out_dict = fn.find_outliers_Z(val)
    totspentVSreord[key] = val[~out_dict]
```


```python
fig = plt.figure(figsize=(9,9))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharey=ax1)

helper = [0, 5, 10, 15]

for key, val in sorted(totspentVSreord.items()):
    if key in helper:
        sns.distplot(val, label=key, ax=ax1).set_title('Total Spent: by Reorder Level sans Outliers (0-15)')
        #plt.legend()
    else:
        sns.distplot(val, label=key, ax=ax2).set_title('Total Spent: by Reorder Level sans Outliers (20-30)')
        #plt.legend()
        
    ax1.legend()
    ax2.legend();
        
plt.tight_layout()


# 2nd visual check (now without outliers)
```

    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    


![png](Images/Mod3_Proj_149_1.png)


> Visually it appears that there are some similarities between the groups I selected arbitrarily by numeric value. For the unit level of 5, there is a high amount that spikes around zero TotalSpent, while there is a long tail created by 15. 

> In the second plot, the categories reflect a similar relationship. Two of the categories have a large grouping around zero TotalSpent while the 30 reorder level creates a tail.

#### Normality?


```python
for key, val in sorted(totspentVSreord.items()):
    stat, p = stats.normaltest(val)
        
    print(f'{key} normal test p-value = {round(p,4)}')
    
    sig = 'is NOT' if p < .05 else 'IS'
    
    print(f'The data {sig} normal.')
```

    0 normal test p-value = 0.0
    The data is NOT normal.
    5 normal test p-value = 0.0
    The data is NOT normal.
    10 normal test p-value = 0.0
    The data is NOT normal.
    15 normal test p-value = 0.0
    The data is NOT normal.
    20 normal test p-value = 0.0
    The data is NOT normal.
    25 normal test p-value = 0.0
    The data is NOT normal.
    30 normal test p-value = 0.0
    The data is NOT normal.
    

> My visual observations were confirmed by stats.normaltest(). The data does not appear to be very close to the standard normal, but with the amount of observations in each group we can move forward here regardless of the failures.


```python
# Quick check for sample sizes

for item in sorted(df4['ReorderLevel'].unique()):
    print(f'Number of instances in {item}: {len(totspentVSreord[item])}')
```

    Number of instances in 0: 703
    Number of instances in 5: 191
    Number of instances in 10: 239
    Number of instances in 15: 264
    Number of instances in 20: 230
    Number of instances in 25: 251
    Number of instances in 30: 232
    

> The requirement to proceed testing while failing normality is recommended at 15, so given these amounts, I will proceed. 

#### Equal Variance?


```python
norm_list4 = []

for key, val in sorted(totspentVSreord.items()):
    norm_list4.append(val)
```


```python
stat, p = stats.levene(*norm_list4)

print(f'Levene test p-value = {round(p,4)}')

sig = 'does NOT' if p < .05 else 'DOES'

print(f'The data {sig} have equal variance.')
```

    Levene test p-value = 0.0
    The data does NOT have equal variance.
    

### Hypothesis Test


```python
stat, p = stats.f_oneway(*norm_list4)

print(f"One-Way ANOVA p-value = {round(p,4)}")

sig = 'IS' if p < .05 else 'is NOT'

print(f'The data {sig} from different populations.')
```

    One-Way ANOVA p-value = 0.0
    The data IS from different populations.
    

> It turns out, despite failing all of the assumptions, we were able to generate a p-value low enough to **Reject the Null Hypothesis ($H_0$)**.

### Post-Hoc Calculations


```python
model4 = pairwise_tukeyhsd(df4['TotalSpent'], df4['ReorderLevel'])
model4.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD,FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>    <th>lower</th>     <th>upper</th>   <th>reject</th>
</tr>
<tr>
     <td>0</td>      <td>5</td>   <td>-477.8568</td> <td>-703.3779</td> <td>-252.3357</td>  <td>True</td> 
</tr>
<tr>
     <td>0</td>     <td>10</td>   <td>-316.1781</td> <td>-524.1813</td> <td>-108.1749</td>  <td>True</td> 
</tr>
<tr>
     <td>0</td>     <td>15</td>    <td>102.942</td>  <td>-96.0326</td>  <td>301.9165</td>   <td>False</td>
</tr>
<tr>
     <td>0</td>     <td>20</td>   <td>-428.3497</td> <td>-638.6571</td> <td>-218.0423</td>  <td>True</td> 
</tr>
<tr>
     <td>0</td>     <td>25</td>   <td>-378.7012</td> <td>-582.4062</td> <td>-174.9963</td>  <td>True</td> 
</tr>
<tr>
     <td>0</td>     <td>30</td>   <td>-223.5092</td> <td>-433.8166</td> <td>-13.2018</td>   <td>True</td> 
</tr>
<tr>
     <td>5</td>     <td>10</td>   <td>161.6786</td>  <td>-107.2305</td> <td>430.5878</td>   <td>False</td>
</tr>
<tr>
     <td>5</td>     <td>15</td>   <td>580.7987</td>  <td>318.8108</td>  <td>842.7867</td>   <td>True</td> 
</tr>
<tr>
     <td>5</td>     <td>20</td>    <td>49.5071</td>  <td>-221.1884</td> <td>320.2025</td>   <td>False</td>
</tr>
<tr>
     <td>5</td>     <td>25</td>    <td>99.1556</td>  <td>-166.4429</td>  <td>364.754</td>   <td>False</td>
</tr>
<tr>
     <td>5</td>     <td>30</td>   <td>254.3476</td>  <td>-16.3479</td>   <td>525.043</td>   <td>False</td>
</tr>
<tr>
    <td>10</td>     <td>15</td>   <td>419.1201</td>  <td>172.0507</td>  <td>666.1895</td>   <td>True</td> 
</tr>
<tr>
    <td>10</td>     <td>20</td>   <td>-112.1716</td> <td>-368.4558</td> <td>144.1127</td>   <td>False</td>
</tr>
<tr>
    <td>10</td>     <td>25</td>   <td>-62.5231</td>  <td>-313.4177</td> <td>188.3715</td>   <td>False</td>
</tr>
<tr>
    <td>10</td>     <td>30</td>    <td>92.6689</td>  <td>-163.6153</td> <td>348.9532</td>   <td>False</td>
</tr>
<tr>
    <td>15</td>     <td>20</td>   <td>-531.2917</td> <td>-780.304</td>  <td>-282.2793</td>  <td>True</td> 
</tr>
<tr>
    <td>15</td>     <td>25</td>   <td>-481.6432</td> <td>-725.105</td>  <td>-238.1814</td>  <td>True</td> 
</tr>
<tr>
    <td>15</td>     <td>30</td>   <td>-326.4512</td> <td>-575.4635</td> <td>-77.4388</td>   <td>True</td> 
</tr>
<tr>
    <td>20</td>     <td>25</td>    <td>49.6485</td>  <td>-203.1597</td> <td>302.4567</td>   <td>False</td>
</tr>
<tr>
    <td>20</td>     <td>30</td>   <td>204.8405</td>  <td>-53.3174</td>  <td>462.9984</td>   <td>False</td>
</tr>
<tr>
    <td>25</td>     <td>30</td>    <td>155.192</td>  <td>-97.6162</td>  <td>408.0002</td>   <td>False</td>
</tr>
</table>



> Significant results:
* 0 reorder level  was shown to be different from all levels *except* 15
* 15 reorder level was shown to be different from **all** other levels 
* No other reorder level proved to be different from each other

> Given this knowledge I will look a bit further into these two areas and see if there is anything worthwhile. 

### Q4 Summary

> Given we know that there *is* an actual difference in the total revenue generated by products at the 15 unit threshold and those that are not reordered until stock is out (0), we can then look to how and why the data could be this way.

> For example, the reason that those products at the 'sold out' threshold may be items ordered less often, in smaller batches, but have a higher average price. Similarly, those items at the 15 unit threshold may be this company's 'staple' products and thus are required to be on hand in a moderate capacity. This information alone can help inform the stakeholders on how to structure their reorder process to reduce waste. 

## Q5: Does the sales representative making the deal have a significant effect on the amount of product ordered? How about total spent?

* $H_0$: The sales person brokering the deal does not have a significant effect on the amount of product in an order nor the total spent.
* $H_1$: The sales person brokering the deal does have a significant effect on the amount of product in an order nor the total spent.

> For this test, I will be using a two-tailed test.

### Initial Query & Feature Engineering


```python
cur.execute('''SELECT od.UnitPrice, od.Quantity, od.Discount,
               ((od.Quantity * od.UnitPrice)*(1 - od.Discount))
               AS TotalSpent, e.Id AS EmployeeId, e.LastName, e.FirstName, e.Title
               FROM OrderDetail od
               JOIN 'Order' o
               ON o.Id = od.OrderId
               JOIN Employee e
               ON o.EmployeeId = e.Id;''')
df5 = pd.DataFrame(cur.fetchall()) 
df5.columns = [i[0] for i in cur.description]
df5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>TotalSpent</th>
      <th>EmployeeId</th>
      <th>LastName</th>
      <th>FirstName</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.0</td>
      <td>12</td>
      <td>0.0</td>
      <td>168.0</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
      <td>Sales Manager</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.8</td>
      <td>10</td>
      <td>0.0</td>
      <td>98.0</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
      <td>Sales Manager</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.8</td>
      <td>5</td>
      <td>0.0</td>
      <td>174.0</td>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
      <td>Sales Manager</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.6</td>
      <td>9</td>
      <td>0.0</td>
      <td>167.4</td>
      <td>6</td>
      <td>Suyama</td>
      <td>Michael</td>
      <td>Sales Representative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42.4</td>
      <td>40</td>
      <td>0.0</td>
      <td>1696.0</td>
      <td>6</td>
      <td>Suyama</td>
      <td>Michael</td>
      <td>Sales Representative</td>
    </tr>
  </tbody>
</table>
</div>




```python
df5.shape
```




    (2155, 8)



### EDA


```python
fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(data=df5, x='EmployeeId', y='Quantity', ci=68, ax=ax)

plt.title('Avg. Quantity: By Salesperson');
```


![png](Images/Mod3_Proj_176_0.png)



```python
sns.catplot(data=df5, x='EmployeeId', y='Quantity', kind='swarm', ci=68)

plt.title('Avg. Quantity: By Salesperson');
```


![png](Images/Mod3_Proj_177_0.png)


> There appears to be some differences, albeit slight, for each employee in regards to quanity sold per order. The swarmplot shows what looks to be a fairly even distribution between each of the employees. The error bars in the barplot indicate that these differences may be 'give-or-take' around the same level. 


```python
fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(data=df5, x='EmployeeId', y='TotalSpent', ci=68, ax=ax)

plt.title('Avg. TotalSpent: By Salesperson');
```


![png](Images/Mod3_Proj_179_0.png)



```python
sns.catplot(data=df5, x='EmployeeId', y='TotalSpent', kind='swarm', ci=68)

plt.title('Avg. TotalSpent: By Salesperson');
```


![png](Images/Mod3_Proj_180_0.png)


> Visually speaking, the TotalSpent category show a lot more variation on the barplot between employees. The swarmplot however, tells a story that appears to indicate not much of a difference between the individual employees. 


```python
quantVSemp = {}
for item in df5['EmployeeId'].unique():
    quantVSemp[item] = df5.groupby('EmployeeId').get_group(item)['Quantity']
```


```python
for item in sorted(df5['EmployeeId'].unique()):
    print(f'Number of instances in {item}: {len(quantVSemp[item])}')
```

    Number of instances in 1: 345
    Number of instances in 2: 241
    Number of instances in 3: 321
    Number of instances in 4: 420
    Number of instances in 5: 117
    Number of instances in 6: 168
    Number of instances in 7: 176
    Number of instances in 8: 260
    Number of instances in 9: 107
    

****************************************************************************


```python
totspentVSemp = {}
for item in df5['EmployeeId'].unique():
    totspentVSemp[item] = df5.groupby('EmployeeId').get_group(item)['TotalSpent']
```


```python
for item in sorted(df5['EmployeeId'].unique()):
    print(f'Number of instances in {item}: {len(totspentVSemp[item])}')
```

    Number of instances in 1: 345
    Number of instances in 2: 241
    Number of instances in 3: 321
    Number of instances in 4: 420
    Number of instances in 5: 117
    Number of instances in 6: 168
    Number of instances in 7: 176
    Number of instances in 8: 260
    Number of instances in 9: 107
    

> Simply put, I will need to use a **One-Way ANOVA** test. This is due to the fact I have *nine* different employees (categories) that I will need to evaluate against each other. 

### Testing Assumptions (One-Way ANOVA)

#### Outliers?


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in sorted(quantVSemp.items()):       
    sns.distplot(val, label=key, ax=ax)

plt.title('Quantity: by Salesperson')
ax.legend();

# Check for overall skew of all data
```


![png](Images/Mod3_Proj_190_0.png)


> It appears that this data is fairly normal, I do not expect too many outliers to be pulled from the right tail.


```python
fig, ax = plt.subplots(figsize=(8,5))

for key, val in sorted(totspentVSemp.items()):       
    sns.distplot(val, label=key, ax=ax)

plt.title('TotalSpent: by Salesperson')
ax.legend();
```


![png](Images/Mod3_Proj_192_0.png)


> This set appears to be *way* more skewed than the quantity data. I expect our outlier remove to cut a decent bit off of the right-side tail.


```python
for key, val in sorted(quantVSemp.items()):
    out_dict = fn.find_outliers_Z(val)
    print(f'There are {out_dict.sum()} Z-outliers (quantity) for employee #{key}.')
    
    #out_dict = fn.find_outliers_IQR(val)
    #print(f'There are {out_dict.sum()} {key} IQR-outliers.')
```

    There are 4 Z-outliers (quantity) for employee #1.
    There are 4 Z-outliers (quantity) for employee #2.
    There are 7 Z-outliers (quantity) for employee #3.
    There are 5 Z-outliers (quantity) for employee #4.
    There are 3 Z-outliers (quantity) for employee #5.
    There are 2 Z-outliers (quantity) for employee #6.
    There are 5 Z-outliers (quantity) for employee #7.
    There are 6 Z-outliers (quantity) for employee #8.
    There are 2 Z-outliers (quantity) for employee #9.
    


```python
for key, val in sorted(totspentVSemp.items()):
    out_dict = fn.find_outliers_Z(val)
    print(f'There are {out_dict.sum()} Z-outliers (total spent) for employee #{key}.')
    
    #out_dict = fn.find_outliers_IQR(val)
    #print(f'There are {out_dict.sum()} {key} IQR-outliers.')
```

    There are 4 Z-outliers (total spent) for employee #1.
    There are 5 Z-outliers (total spent) for employee #2.
    There are 4 Z-outliers (total spent) for employee #3.
    There are 7 Z-outliers (total spent) for employee #4.
    There are 3 Z-outliers (total spent) for employee #5.
    There are 3 Z-outliers (total spent) for employee #6.
    There are 5 Z-outliers (total spent) for employee #7.
    There are 5 Z-outliers (total spent) for employee #8.
    There are 2 Z-outliers (total spent) for employee #9.
    

> I am surprised to see that the TotalSpent data had about the amount of outliers despite what the distplots show of it. Perhaps there are only a few data points dragging it out that far. I am interested to see what the data looks like after removal. 


```python
# Z-score method removal 

for key, val in quantVSemp.items():
    out_dict = fn.find_outliers_Z(val)
    quantVSemp[key] = val[~out_dict]
```


```python
fig = plt.figure(figsize=(9,9))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharey=ax1)

helper2 = [1, 2, 3, 4]

for key, val in sorted(quantVSemp.items()):
    if key in helper2:
        sns.distplot(val, label=key, ax=ax1).set_title('Quantity: by Salesperson sans Outliers (1-5)')
    else:
        sns.distplot(val, label=key, ax=ax2).set_title('Quantity: by Salesperson sans Outliers (6-9)')
        
    ax1.legend()
    ax2.legend();
        
plt.tight_layout()
```

    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    


![png](Images/Mod3_Proj_198_1.png)


> Looking at these graphs, it appears that employees 5, 6, and 8 have a higher percentage of their orders being quantities less than 20. Otherwise both groups seem to be about the same.


```python
for key, val in totspentVSemp.items():
    out_dict = fn.find_outliers_Z(val)
    totspentVSemp[key] = val[~out_dict]
```


```python
fig = plt.figure(figsize=(9,9))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharey=ax1)

helper2 = [1, 2, 3, 4]

for key, val in sorted(totspentVSemp.items()):
    if key in helper2:
        sns.distplot(val, label=key, ax=ax1).set_title('TotalSpent: by Salesperson sans Outliers (1-5)')
    else:
        sns.distplot(val, label=key, ax=ax2).set_title('TotalSpent: by Salesperson sans Outliers (6-9)')
        
    ax1.legend()
    ax2.legend();
        
plt.tight_layout()
```

    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    No handles with labels found to put in legend.
    


![png](Images/Mod3_Proj_201_1.png)


> Visually, these two groups appear to be quite similar. The second group just has a bit longer of a right-tail due to a few instances.

#### Normality?


```python
for key, val in sorted(quantVSemp.items()):
    stat, p = stats.normaltest(val)
        
    print(f'Employee #{key} normal test p-value = {round(p,4)}')
    
    sig = 'is NOT' if p < .05 else 'IS'
    
    print(f'The data {sig} normal.')
```

    Employee #1 normal test p-value = 0.0
    The data is NOT normal.
    Employee #2 normal test p-value = 0.0
    The data is NOT normal.
    Employee #3 normal test p-value = 0.0
    The data is NOT normal.
    Employee #4 normal test p-value = 0.0
    The data is NOT normal.
    Employee #5 normal test p-value = 0.0
    The data is NOT normal.
    Employee #6 normal test p-value = 0.0
    The data is NOT normal.
    Employee #7 normal test p-value = 0.0
    The data is NOT normal.
    Employee #8 normal test p-value = 0.0
    The data is NOT normal.
    Employee #9 normal test p-value = 0.0115
    The data is NOT normal.
    

> Although it Employee #9 had a p-value slightly smaller than our alpha, all groups have been determined to be **non-normal**. I will double-check that we have large enough sample sizes to move forward.


```python
for item in sorted(df5['EmployeeId'].unique()):
    print(f'Number of instances in Employee #{item}: {len(quantVSemp[item])}')
```

    Number of instances in Employee #1: 341
    Number of instances in Employee #2: 237
    Number of instances in Employee #3: 314
    Number of instances in Employee #4: 415
    Number of instances in Employee #5: 114
    Number of instances in Employee #6: 166
    Number of instances in Employee #7: 171
    Number of instances in Employee #8: 254
    Number of instances in Employee #9: 105
    


```python
for key, val in sorted(totspentVSemp.items()):
    stat, p = stats.normaltest(val)
        
    print(f'Employee #{key} normal test p-value = {round(p,4)}')
    
    sig = 'is NOT' if p < .05 else 'IS'
    
    print(f'The data {sig} normal.')
```

    Employee #1 normal test p-value = 0.0
    The data is NOT normal.
    Employee #2 normal test p-value = 0.0
    The data is NOT normal.
    Employee #3 normal test p-value = 0.0
    The data is NOT normal.
    Employee #4 normal test p-value = 0.0
    The data is NOT normal.
    Employee #5 normal test p-value = 0.0
    The data is NOT normal.
    Employee #6 normal test p-value = 0.0
    The data is NOT normal.
    Employee #7 normal test p-value = 0.0
    The data is NOT normal.
    Employee #8 normal test p-value = 0.0
    The data is NOT normal.
    Employee #9 normal test p-value = 0.0
    The data is NOT normal.
    

> All of the groups are **non-normal**. I need to check the sample size.


```python
for item in sorted(df5['EmployeeId'].unique()):
    print(f'Number of instances in Employee #{item}: {len(totspentVSemp[item])}')
```

    Number of instances in Employee #1: 341
    Number of instances in Employee #2: 236
    Number of instances in Employee #3: 317
    Number of instances in Employee #4: 413
    Number of instances in Employee #5: 114
    Number of instances in Employee #6: 165
    Number of instances in Employee #7: 171
    Number of instances in Employee #8: 255
    Number of instances in Employee #9: 105
    

> Both the Quantity and TotalSpent data have enough in the groups to ignore the assumption of normality for our ANOVA.

#### Equal Variance?


```python
norm_list5 = []

for key, val in sorted(quantVSemp.items()):
    norm_list5.append(val)
```


```python
stat, p = stats.levene(*norm_list5)

print(f'Levene test p-value = {round(p,4)}')

sig = 'does NOT' if p < .05 else 'DOES'

print(f'The data {sig} have equal variance.')
```

    Levene test p-value = 0.0939
    The data DOES have equal variance.
    

****************************************************************************


```python
norm_list6 = []

for key, val in sorted(totspentVSemp.items()):
    norm_list6.append(val)
```


```python
stat, p = stats.levene(*norm_list6)

print(f'Levene test p-value = {round(p,4)}')

sig = 'does NOT' if p < .05 else 'DOES'

print(f'The data {sig} have equal variance.')
```

    Levene test p-value = 0.0143
    The data does NOT have equal variance.
    

> Surprisingly, the data subsets concerning quantity does have equal variance while the other does not. I will *need* to do a Tukey's test on the total spent due to it failing the Levene's test.

### Hypothesis Test


```python
stat, p = stats.f_oneway(*norm_list5)

print(f"One-Way ANOVA p-value = {round(p,4)}")

sig = 'IS' if p < .05 else 'is NOT'

print(f'The data {sig} from different populations.')
```

    One-Way ANOVA p-value = 0.1783
    The data is NOT from different populations.
    

> The One-Way ANOVA indicates that I have **failed to reject the Null Hypothesis ($H_0$)**. There is not enough evidence to say that any given employee has an effect upon the quantity sold in an order.


```python
stat, p = stats.f_oneway(*norm_list6)

print(f"One-Way ANOVA p-value = {round(p,4)}")

sig = 'IS' if p < .05 else 'is NOT'

print(f'The data {sig} from different populations.')
```

    One-Way ANOVA p-value = 0.0262
    The data IS from different populations.
    

> In the case of the total amount spent on an order, the One-Way ANOVA allows me to **reject the Null Hypothesis ($H_0$)**. The data does in fact suggest a statistical difference between employees when is comes to the total bill.

### Post-Hoc Calculations


```python
model5 = pairwise_tukeyhsd(df5['Quantity'], df5['EmployeeId'])
model5.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD,FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>   <th>lower</th>   <th>upper</th>  <th>reject</th>
</tr>
<tr>
     <td>1</td>      <td>2</td>     <td>2.481</td>   <td>-2.4718</td> <td>7.4338</td>   <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>3</td>    <td>1.8176</td>   <td>-2.7574</td> <td>6.3926</td>   <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>4</td>    <td>0.6851</td>   <td>-3.6015</td> <td>4.9717</td>   <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>5</td>    <td>3.3052</td>   <td>-3.0063</td> <td>9.6168</td>   <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>6</td>    <td>-1.6494</td>  <td>-7.1997</td> <td>3.9008</td>   <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>7</td>    <td>3.7997</td>   <td>-1.6651</td> <td>9.2645</td>   <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>8</td>    <td>0.0988</td>   <td>-4.7462</td> <td>4.9439</td>   <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>9</td>    <td>2.3098</td>   <td>-4.2183</td> <td>8.8379</td>   <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>3</td>    <td>-0.6634</td>  <td>-5.6918</td> <td>4.3649</td>   <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>4</td>    <td>-1.7959</td>  <td>-6.5634</td> <td>2.9715</td>   <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>5</td>    <td>0.8242</td>   <td>-5.8233</td> <td>7.4717</td>   <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>6</td>    <td>-4.1304</td> <td>-10.0599</td>  <td>1.799</td>   <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>7</td>    <td>1.3187</td>   <td>-4.5308</td> <td>7.1682</td>   <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>8</td>    <td>-2.3822</td>  <td>-7.6574</td> <td>2.8931</td>   <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>9</td>    <td>-0.1712</td>  <td>-7.0246</td> <td>6.6822</td>   <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>4</td>    <td>-1.1325</td>  <td>-5.5062</td> <td>3.2412</td>   <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>5</td>    <td>1.4877</td>   <td>-4.8834</td> <td>7.8587</td>   <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>6</td>    <td>-3.467</td>   <td>-9.0848</td> <td>2.1508</td>   <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>7</td>    <td>1.9821</td>   <td>-3.5512</td> <td>7.5155</td>   <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>8</td>    <td>-1.7188</td>  <td>-6.641</td>  <td>3.2035</td>   <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>9</td>    <td>0.4922</td>   <td>-6.0934</td> <td>7.0778</td>   <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>5</td>    <td>2.6201</td>   <td>-3.547</td>  <td>8.7873</td>   <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>6</td>    <td>-2.3345</td>   <td>-7.72</td>   <td>3.051</td>   <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>7</td>    <td>3.1146</td>   <td>-2.1828</td>  <td>8.412</td>   <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>8</td>    <td>-0.5863</td>  <td>-5.2417</td> <td>4.0692</td>   <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>9</td>    <td>1.6247</td>   <td>-4.7639</td> <td>8.0133</td>   <td>False</td>
</tr>
<tr>
     <td>5</td>      <td>6</td>    <td>-4.9547</td> <td>-12.0585</td> <td>2.1492</td>   <td>False</td>
</tr>
<tr>
     <td>5</td>      <td>7</td>    <td>0.4945</td>   <td>-6.5428</td> <td>7.5317</td>   <td>False</td>
</tr>
<tr>
     <td>5</td>      <td>8</td>    <td>-3.2064</td>  <td>-9.774</td>  <td>3.3612</td>   <td>False</td>
</tr>
<tr>
     <td>5</td>      <td>9</td>    <td>-0.9954</td>  <td>-8.8869</td>  <td>6.896</td>   <td>False</td>
</tr>
<tr>
     <td>6</td>      <td>7</td>    <td>5.4491</td>   <td>-0.9142</td> <td>11.8125</td>  <td>False</td>
</tr>
<tr>
     <td>6</td>      <td>8</td>    <td>1.7483</td>   <td>-4.0916</td> <td>7.5881</td>   <td>False</td>
</tr>
<tr>
     <td>6</td>      <td>9</td>    <td>3.9592</td>   <td>-3.3377</td> <td>11.2561</td>  <td>False</td>
</tr>
<tr>
     <td>7</td>      <td>8</td>    <td>-3.7009</td>  <td>-9.4595</td> <td>2.0577</td>   <td>False</td>
</tr>
<tr>
     <td>7</td>      <td>9</td>    <td>-1.4899</td>  <td>-8.722</td>  <td>5.7422</td>   <td>False</td>
</tr>
<tr>
     <td>8</td>      <td>9</td>     <td>2.211</td>   <td>-4.565</td>   <td>8.987</td>   <td>False</td>
</tr>
</table>



> There are no groups that show significantly different.


```python
model6 = pairwise_tukeyhsd(df5['TotalSpent'], df5['EmployeeId'])
model6.summary()
```




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD,FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>    <th>lower</th>     <th>upper</th>  <th>reject</th>
</tr>
<tr>
     <td>1</td>      <td>2</td>   <td>134.1944</td>  <td>-117.9953</td> <td>386.3841</td>  <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>3</td>    <td>74.9821</td>  <td>-157.9727</td> <td>307.9368</td>  <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>4</td>    <td>-2.3316</td>  <td>-220.6011</td> <td>215.9379</td>  <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>5</td>    <td>31.1346</td>  <td>-290.2426</td> <td>352.5118</td>  <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>6</td>   <td>-116.8745</td> <td>-399.4868</td> <td>165.7378</td>  <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>7</td>   <td>150.9404</td>  <td>-127.3188</td> <td>429.1996</td>  <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>8</td>   <td>-68.9018</td>  <td>-315.6068</td> <td>177.8032</td>  <td>False</td>
</tr>
<tr>
     <td>1</td>      <td>9</td>   <td>165.6717</td>  <td>-166.7309</td> <td>498.0742</td>  <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>3</td>   <td>-59.2123</td>  <td>-315.2499</td> <td>196.8252</td>  <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>4</td>   <td>-136.526</td>  <td>-379.2787</td> <td>106.2267</td>  <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>5</td>   <td>-103.0598</td> <td>-441.5426</td>  <td>235.423</td>  <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>6</td>   <td>-251.0689</td> <td>-552.9911</td>  <td>50.8533</td>  <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>7</td>    <td>16.746</td>   <td>-281.1054</td> <td>314.5975</td>  <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>8</td>   <td>-203.0962</td> <td>-471.705</td>   <td>65.5126</td>  <td>False</td>
</tr>
<tr>
     <td>2</td>      <td>9</td>    <td>31.4773</td>  <td>-317.4909</td> <td>380.4454</td>  <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>4</td>   <td>-77.3137</td>  <td>-300.0179</td> <td>145.3906</td>  <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>5</td>   <td>-43.8475</td>  <td>-368.253</td>   <td>280.558</td>  <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>6</td>   <td>-191.8566</td> <td>-477.9078</td>  <td>94.1946</td>  <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>7</td>    <td>75.9584</td>  <td>-205.7929</td> <td>357.7096</td>  <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>8</td>   <td>-143.8839</td> <td>-394.521</td>  <td>106.7533</td>  <td>False</td>
</tr>
<tr>
     <td>3</td>      <td>9</td>    <td>90.6896</td>  <td>-244.6417</td> <td>426.0208</td>  <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>5</td>    <td>33.4662</td>  <td>-280.5602</td> <td>347.4926</td>  <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>6</td>   <td>-114.5429</td> <td>-388.7672</td> <td>159.6814</td>  <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>7</td>    <td>153.272</td>  <td>-116.4638</td> <td>423.0079</td>  <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>8</td>   <td>-66.5702</td>   <td>-303.62</td>  <td>170.4797</td>  <td>False</td>
</tr>
<tr>
     <td>4</td>      <td>9</td>   <td>168.0033</td>  <td>-157.2977</td> <td>493.3042</td>  <td>False</td>
</tr>
<tr>
     <td>5</td>      <td>6</td>   <td>-148.0091</td> <td>-509.7282</td>  <td>213.71</td>   <td>False</td>
</tr>
<tr>
     <td>5</td>      <td>7</td>   <td>119.8058</td>  <td>-238.5225</td> <td>478.1341</td>  <td>False</td>
</tr>
<tr>
     <td>5</td>      <td>8</td>   <td>-100.0364</td> <td>-434.4528</td> <td>234.3801</td>  <td>False</td>
</tr>
<tr>
     <td>5</td>      <td>9</td>   <td>134.5371</td>  <td>-267.2868</td>  <td>536.361</td>  <td>False</td>
</tr>
<tr>
     <td>6</td>      <td>7</td>    <td>267.815</td>  <td>-56.1998</td>  <td>591.8297</td>  <td>False</td>
</tr>
<tr>
     <td>6</td>      <td>8</td>    <td>47.9727</td>  <td>-249.3836</td>  <td>345.329</td>  <td>False</td>
</tr>
<tr>
     <td>6</td>      <td>9</td>   <td>282.5462</td>  <td>-89.0031</td>  <td>654.0954</td>  <td>False</td>
</tr>
<tr>
     <td>7</td>      <td>8</td>   <td>-219.8422</td> <td>-513.0644</td>  <td>73.3799</td>  <td>False</td>
</tr>
<tr>
     <td>7</td>      <td>9</td>    <td>14.7312</td>  <td>-353.5178</td> <td>382.9802</td>  <td>False</td>
</tr>
<tr>
     <td>8</td>      <td>9</td>   <td>234.5735</td>  <td>-110.4519</td> <td>579.5988</td>  <td>False</td>
</tr>
</table>



> Despite getting a significant result with the One-Way ANOVA, the Tukey's test was unable to detect any significance between any groups within the dataset.

### Q5 Summary

> The first part of my hypothesis, although promising, failed to produce significant results. Alone this result shows that currently all of the sales team is moving a consistent amount of product and on average *do not* out perform one another. 

> The second part has yielded interesting results. This leads me to believe that I potentially got a Type I Error for the Total Spent data. I would need more data and/or run more models to verify my results.

## Results

**Hypothesis Summary**

* **Q1**: *Rejected $H_0$*. Discount levels of 5%, 15%, 20%, and 25% were shown to have an effect on the average quantity.

* **Q2**: *Failed to reject $H_0$*. There was not enough evidence to suggest discounting has an effect on the total amount spent on an order. This is also true amongst each of the discount levels.

* **Q3**: *Failed to reject $H_0$*. The data was unable to demonstrate that the region a product was sold in has an effect on the quantity of product in an order. This is also true between each of the regions.

* **Q4**: *Rejected $H_0$*. The data indicated that products they waited until sold out to reorder and/or when 15 units were left showed a difference in the total amount spent on orders they were included in.

* **Q5**:
    * **Quantity**: *Failed to reject $H_0$*. The statistical test was unable to confirm that the employee involved in a given sale has any significant effect on the amount of product purchased.
    * **Total Spent**: *Rejected $H_0$*. Our One-Way ANOVA test allowed us to reject the null, while our Tukey's test could not show a specific group to be significantly different than another.


```python

```
