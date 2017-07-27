# divergence
This code allows you to create divergence matrices from Wikum et al

#Example

```python
    df # a dataframe that has analytes as columns and observations as
index
    training_set # a list of observations to train on from df index
    test_set # a list of observations in the result from df index

    from divergence import *

    #set parameters
    d = Divergence(df.loc[training_set], lower=.025, upper=.975,
quantize=True)
    #transform test set
    div_df = d.transform(df.loc[test_set], quantize=True)
    #div_df is your divergence matrix
```
