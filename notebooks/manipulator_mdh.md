```python
import os
import sympy as sp
import numpy as np
from IPython.display import display
import pandas as pd
```


```python
PARAMS_DIR = os.path.join(os.getcwd(), "../mdh_params")
```


```python
# Define the symbolic variables
alpha, a, theta, d = sp.symbols('alpha a theta d')
```


```python
def mdh_tf(alpha, a, theta, d):
    """
    Compute the Modified Denavit-Hartenberg transformation matrix.

    Parameters:
    alpha : float
        The twist angle (in radians).
    a : float
        The link length.
    theta : float
        The joint angle (in radians).
    d : float
        The link offset.

    Returns:
    tf : sympy.Matrix
        The 4x4 transformation matrix.
    """

    tf = sp.Matrix([
        [sp.cos(theta),               -sp.sin(theta),              0,               a               ],   # type: ignore
        [sp.sin(theta)*sp.cos(alpha), sp.cos(theta)*sp.cos(alpha), -sp.sin(alpha),  -d*sp.sin(alpha)],   # type: ignore
        [sp.sin(alpha)*sp.sin(theta), sp.sin(alpha)*sp.cos(theta), sp.cos(alpha),   d*sp.cos(alpha) ],   # type: ignore
        [0,                           0,                           0,               1               ]
    ])
    
    return tf
```


```python
param_files = os.listdir(PARAMS_DIR)
display(param_files)
```


    ['scara.csv',
     'cylindrical.csv',
     'spherical.csv',
     'articulated.csv',
     'cartesian.csv']



```python
def parse_angle_deg(val):
    """Converts numbers or string expressions in degrees to SymPy radians."""
    # Parse strings (like 'theta1' or 'theta1 + 180') or keep numbers/symbols as-is
    expr = sp.sympify(val)
    
    # If it's a pure number (int/float), convert degrees to radians
    if expr.is_number:
        return sp.rad(expr)
    
    # If it's a symbolic expression (e.g., theta1 + 90), convert any numeric constants to radians
    # Note: If your expressions are just 'theta1', this leaves 'theta1' untouched as a symbol
    return expr.replace(lambda x: x.is_Number and x != 0, lambda x: sp.rad(x))
```


```python
def compute_mdh_transformations(mdh_params):
    """
    Compute the MDH transformation matrices for a given parameter file.

    Parameters:
    mdh_params : DataFrame
        The DataFrame containing the MDH parameters.

    Returns:
    transformations : list of sympy.Matrix
        A list of 4x4 transformation matrices for each link.
    """

    transformations = []
    
    for index, row in mdh_params.iterrows():
        alpha_val = parse_angle_deg(row['alpha'])
        a_val = sp.sympify(row['a'])
        theta_val = parse_angle_deg(row['theta'])
        d_val = sp.sympify(row['d'])
        
        tf = mdh_tf(alpha_val, a_val, theta_val, d_val)
        transformations.append(tf)

        # Display each transformation matrix
        label = sp.MatrixSymbol(f'A{index+1}', 4, 4)
        eq = sp.Eq(label, tf)
        display(eq)
    
    return transformations
```


```python
for param_file in param_files:
    manipulator_name = param_file.split('.')[0]
    print(f"Manipulator: {manipulator_name.capitalize()}")
    mdh_params = pd.read_csv(f"{PARAMS_DIR}/{param_file}")
    display(mdh_params)

    tf_matrices = compute_mdh_transformations(mdh_params)

    T_0_n = sp.eye(4)  # Initialize the overall transformation matrix as identity
    for i, tf in enumerate(tf_matrices):
        T_0_n = sp.simplify(T_0_n * tf)  # Multiply the transformation matrices
    
    label = sp.MatrixSymbol(f'{{}}^{{{0}}}T_{{{i+1}}}', 4, 4)
    eq = sp.Eq(label, T_0_n)
    display(eq)
    print("=" * 100 + "\n" * 2)

```

Manipulator: Scara



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
      <th>alpha</th>
      <th>a</th>
      <th>theta</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>theta1</td>
      <td>d1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>a2</td>
      <td>theta2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>a3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>180</td>
      <td>0</td>
      <td>0</td>
      <td>d4</td>
    </tr>
  </tbody>
</table>
</div>



$\displaystyle A_{1} = \left[\begin{matrix}\cos{\left(\theta_{1} \right)} & - \sin{\left(\theta_{1} \right)} & 0 & 0\\\sin{\left(\theta_{1} \right)} & \cos{\left(\theta_{1} \right)} & 0 & 0\\0 & 0 & 1 & d_{1}\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{2} = \left[\begin{matrix}\cos{\left(\theta_{2} \right)} & - \sin{\left(\theta_{2} \right)} & 0 & a_{2}\\\sin{\left(\theta_{2} \right)} & \cos{\left(\theta_{2} \right)} & 0 & 0\\0 & 0 & 1 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{3} = \left[\begin{matrix}1 & 0 & 0 & a_{3}\\0 & 1 & 0 & 0\\0 & 0 & 1 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{4} = \left[\begin{matrix}1 & 0 & 0 & 0\\0 & -1 & 0 & 0\\0 & 0 & -1 & - d_{4}\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle {}^{0}T_{4} = \left[\begin{matrix}\cos{\left(\theta_{1} + \theta_{2} \right)} & \sin{\left(\theta_{1} + \theta_{2} \right)} & 0 & a_{2} \cos{\left(\theta_{1} \right)} + a_{3} \cos{\left(\theta_{1} + \theta_{2} \right)}\\\sin{\left(\theta_{1} + \theta_{2} \right)} & - \cos{\left(\theta_{1} + \theta_{2} \right)} & 0 & a_{2} \sin{\left(\theta_{1} \right)} + a_{3} \sin{\left(\theta_{1} + \theta_{2} \right)}\\0 & 0 & -1 & d_{1} - d_{4}\\0 & 0 & 0 & 1\end{matrix}\right]$


====================================================================================================
    
    
Manipulator: Cylindrical



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
      <th>alpha</th>
      <th>a</th>
      <th>theta</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>theta1</td>
      <td>d1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>d2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-90</td>
      <td>0</td>
      <td>0</td>
      <td>d3</td>
    </tr>
  </tbody>
</table>
</div>



$\displaystyle A_{1} = \left[\begin{matrix}\cos{\left(\theta_{1} \right)} & - \sin{\left(\theta_{1} \right)} & 0 & 0\\\sin{\left(\theta_{1} \right)} & \cos{\left(\theta_{1} \right)} & 0 & 0\\0 & 0 & 1 & d_{1}\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{2} = \left[\begin{matrix}1 & 0 & 0 & 0\\0 & 1 & 0 & 0\\0 & 0 & 1 & d_{2}\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{3} = \left[\begin{matrix}1 & 0 & 0 & 0\\0 & 0 & 1 & d_{3}\\0 & -1 & 0 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle {}^{0}T_{3} = \left[\begin{matrix}\cos{\left(\theta_{1} \right)} & 0 & - \sin{\left(\theta_{1} \right)} & - d_{3} \sin{\left(\theta_{1} \right)}\\\sin{\left(\theta_{1} \right)} & 0 & \cos{\left(\theta_{1} \right)} & d_{3} \cos{\left(\theta_{1} \right)}\\0 & -1 & 0 & d_{1} + d_{2}\\0 & 0 & 0 & 1\end{matrix}\right]$


====================================================================================================
    
    
Manipulator: Spherical



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
      <th>alpha</th>
      <th>a</th>
      <th>theta</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>theta1</td>
      <td>d1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-90</td>
      <td>0</td>
      <td>theta2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>d3</td>
    </tr>
  </tbody>
</table>
</div>



$\displaystyle A_{1} = \left[\begin{matrix}\cos{\left(\theta_{1} \right)} & - \sin{\left(\theta_{1} \right)} & 0 & 0\\\sin{\left(\theta_{1} \right)} & \cos{\left(\theta_{1} \right)} & 0 & 0\\0 & 0 & 1 & d_{1}\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{2} = \left[\begin{matrix}\cos{\left(\theta_{2} \right)} & - \sin{\left(\theta_{2} \right)} & 0 & 0\\0 & 0 & 1 & 0\\- \sin{\left(\theta_{2} \right)} & - \cos{\left(\theta_{2} \right)} & 0 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{3} = \left[\begin{matrix}1 & 0 & 0 & 0\\0 & 0 & -1 & - d_{3}\\0 & 1 & 0 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle {}^{0}T_{3} = \left[\begin{matrix}\cos{\left(\theta_{1} \right)} \cos{\left(\theta_{2} \right)} & - \sin{\left(\theta_{1} \right)} & \sin{\left(\theta_{2} \right)} \cos{\left(\theta_{1} \right)} & d_{3} \sin{\left(\theta_{2} \right)} \cos{\left(\theta_{1} \right)}\\\sin{\left(\theta_{1} \right)} \cos{\left(\theta_{2} \right)} & \cos{\left(\theta_{1} \right)} & \sin{\left(\theta_{1} \right)} \sin{\left(\theta_{2} \right)} & d_{3} \sin{\left(\theta_{1} \right)} \sin{\left(\theta_{2} \right)}\\- \sin{\left(\theta_{2} \right)} & 0 & \cos{\left(\theta_{2} \right)} & d_{1} + d_{3} \cos{\left(\theta_{2} \right)}\\0 & 0 & 0 & 1\end{matrix}\right]$


====================================================================================================
    
    
Manipulator: Articulated



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
      <th>alpha</th>
      <th>a</th>
      <th>theta</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>theta1</td>
      <td>d1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-90</td>
      <td>0</td>
      <td>theta2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>a2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>a3</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



$\displaystyle A_{1} = \left[\begin{matrix}\cos{\left(\theta_{1} \right)} & - \sin{\left(\theta_{1} \right)} & 0 & 0\\\sin{\left(\theta_{1} \right)} & \cos{\left(\theta_{1} \right)} & 0 & 0\\0 & 0 & 1 & d_{1}\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{2} = \left[\begin{matrix}\cos{\left(\theta_{2} \right)} & - \sin{\left(\theta_{2} \right)} & 0 & 0\\0 & 0 & 1 & 0\\- \sin{\left(\theta_{2} \right)} & - \cos{\left(\theta_{2} \right)} & 0 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{3} = \left[\begin{matrix}1 & 0 & 0 & a_{2}\\0 & 1 & 0 & 0\\0 & 0 & 1 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle A_{4} = \left[\begin{matrix}1 & 0 & 0 & a_{3}\\0 & 1 & 0 & 0\\0 & 0 & 1 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



$\displaystyle {}^{0}T_{4} = \left[\begin{matrix}\cos{\left(\theta_{1} \right)} \cos{\left(\theta_{2} \right)} & - \sin{\left(\theta_{2} \right)} \cos{\left(\theta_{1} \right)} & - \sin{\left(\theta_{1} \right)} & \left(a_{2} + a_{3}\right) \cos{\left(\theta_{1} \right)} \cos{\left(\theta_{2} \right)}\\\sin{\left(\theta_{1} \right)} \cos{\left(\theta_{2} \right)} & - \sin{\left(\theta_{1} \right)} \sin{\left(\theta_{2} \right)} & \cos{\left(\theta_{1} \right)} & \left(a_{2} + a_{3}\right) \sin{\left(\theta_{1} \right)} \cos{\left(\theta_{2} \right)}\\- \sin{\left(\theta_{2} \right)} & - \cos{\left(\theta_{2} \right)} & 0 & - a_{2} \sin{\left(\theta_{2} \right)} - a_{3} \sin{\left(\theta_{2} \right)} + d_{1}\\0 & 0 & 0 & 1\end{matrix}\right]$


    ====================================================================================================
    
    
    Manipulator: Cartesian



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
      <th>alpha</th>
      <th>a</th>
      <th>theta</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



$\displaystyle {}^{0}T_{4} = \left[\begin{matrix}1 & 0 & 0 & 0\\0 & 1 & 0 & 0\\0 & 0 & 1 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$


    ====================================================================================================
    
    



```python
!jupyter nbconvert --to markdown manipulator_mdh.ipynb

```


```python

```
