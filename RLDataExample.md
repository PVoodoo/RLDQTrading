

```python
import pandas as pd
import numpy as np

```


```python
df = pd.read_csv("RLDataForCL60.csv",sep=';', parse_dates=[0])
```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UTC0</th>
      <th>Input0</th>
      <th>134_0</th>
      <th>76_0</th>
      <th>125_0 (Info_to_ignore: TickSize: 0.01 PointValue: 1000)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-04 11:00:00</td>
      <td>48.69</td>
      <td>0.478261</td>
      <td>71.577411</td>
      <td>2.311410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-04 12:00:00</td>
      <td>48.88</td>
      <td>0.434783</td>
      <td>73.889657</td>
      <td>2.710653</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-04 13:00:00</td>
      <td>48.63</td>
      <td>0.391304</td>
      <td>66.252309</td>
      <td>2.185333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-04 13:47:00</td>
      <td>48.79</td>
      <td>0.357246</td>
      <td>68.496609</td>
      <td>2.521538</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-04 14:29:00</td>
      <td>48.83</td>
      <td>0.326812</td>
      <td>69.050739</td>
      <td>2.605589</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    UTC0                                                       datetime64[ns]
    Input0                                                            float64
    134_0                                                             float64
    76_0                                                              float64
    125_0 (Info_to_ignore: TickSize: 0.01 PointValue: 1000)           float64
    dtype: object



Or use directly numpy as dates are not needed


```python
data = np.genfromtxt("RLDataForCL60.csv",delimiter=';',skip_header=1,  dtype="float_")[:,1:]  # or  use usecols=(1,2,3,..) and no [:,1:]
```


```python
data[0:5]
```




    array([[48.69      ,  0.47826087, 71.57741085,  2.31140996],
           [48.88      ,  0.43478261, 73.88965726,  2.7106535 ],
           [48.63      ,  0.39130435, 66.25230913,  2.18533305],
           [48.79      ,  0.35724638, 68.49660906,  2.52153814],
           [48.83      ,  0.32681159, 69.05073924,  2.60558941]])



Scaling next, price not! which is used to calculate reward not a feature, if feature add separately MLIndicatorI 13 = Close
Sacling can be other too like [-1 - 1] 


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0.1, 1))

```


```python
features = data[:,1:] # price removed, column 1
```


```python
scaler.fit(features)   # this need to be put to Dim 2 not to dim 3 values next
scaledFeatures = scaler.transform(features)
```


```python
scaledFeatures[0:5]  # just head
```




    array([[0.54864048, 0.87527158, 0.71767151],
           [0.50785498, 0.90803152, 0.75684043],
           [0.46706949, 0.79982548, 0.70530238],
           [0.43512085, 0.83162275, 0.73828673],
           [0.406571  , 0.83947367, 0.74653282]])




```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
 df.plot();
```


![png](output_13_0.png)



```python
plt.plot(scaledFeatures[:200]) # part
plt.show()
```


![png](output_14_0.png)

