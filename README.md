
# Simple Linear Regression – Beginner Notes

---

## 1. What is Simple Linear Regression?

**Simple Linear Regression** is a way to **predict one outcome (y)** using **one input (x)**, assuming a straight-line (linear) relationship between the two.

### Real-Life Analogy:

Imagine you're trying to guess someone’s **weight** just by knowing their **height**. If generally, taller people weigh more, then drawing a straight line through a scatter plot of height and weight gives a way to **predict weight for new people**.

---

## 2. The Formula of Simple Linear Regression

The general equation is:

$$
y = \beta_0 + \beta_1 x
$$

Where:

* $y$: predicted value (e.g., weight)
* $x$: input variable (e.g., height)
* $\beta_0$: intercept (starting point when $x = 0$)
* $\beta_1$: slope (how much $y$ increases when $x$ increases by 1)

---

### Example:

Let’s say your model is:

$$
\text{Weight} = 50 + 1.2 \cdot \text{Height}
$$

Then:

* When height = 160 cm → predicted weight = $50 + 1.2 \times 160 = 242$ kg

---

## 3. Visual Intuition (ASCII Style)

```
    Weight
     ▲
 250 |                      *
     |
 200 |              *    
     |
 150 |         *
     |
 100 |    *          
     |
     └────────────────────────► Height
         140 150 160 170 180
```

The red line through the points is the **regression line** — the best fit through the cloud of dots.

---

## 4. Real Dataset Example: Student Study Hours vs Exam Scores

### Dataset: [Kaggle - Student Scores](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

We use a version with:

* Input: Hours studied (x)
* Output: Exam score (y)

---

## 5. Python Code Implementation

### Step 1: Setup

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

### Step 2: Data

```python
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [35, 40, 50, 55, 60, 65, 70, 78, 85, 95]
}
df = pd.DataFrame(data)
```

### Step 3: Visualize the Data

```python
plt.scatter(df['Hours'], df['Scores'], color='blue')
plt.title('Study Hours vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.grid(True)
plt.show()
```

---

### Step 4: Fit the Model

```python
X = df[['Hours']]
y = df['Scores']

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
```

---

### Step 5: Predict and Plot

```python
df['Predicted'] = model.predict(X)

plt.scatter(df['Hours'], df['Scores'], label='Actual', color='blue')
plt.plot(df['Hours'], df['Predicted'], label='Regression Line', color='red')
plt.title('Simple Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 6. Understanding Errors & Model Performance

### A. **Residuals**

$$
\text{Residual} = \text{Actual} - \text{Predicted}
$$

* Residuals are the **tiny differences** between what the model guessed and what actually happened.
* **Smaller residuals** mean better predictions.

### Analogy:

You guessed your friend will score 85, but she scores 90. The **residual = -5**. That’s how wrong you were.

---

### B. **Sum of Squared Errors (SSE)**

$$
\text{SSE} = \sum (y_i - \hat{y}_i)^2
$$

* It **adds up all residuals, squared**.
* Squaring makes all errors positive and penalizes large mistakes.

### Analogy:

SSE is like the **total penalty** for all your prediction mistakes — the bigger the misses, the more you pay.

---

### C. **Mean Absolute Error (MAE)**

$$
\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|
$$

* It’s the **average size of the error**, ignoring whether it’s too high or too low.
* Units are the same as the output.

### Analogy:

If you predict delivery times and you’re off by 3, 5, and 2 minutes, your **MAE = (3 + 5 + 2)/3 = 3.33 mins**
It’s like saying, **“On average, I’m 3.33 minutes wrong.”**

---

### D. **Mean Squared Error (MSE)**

$$
\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
$$

* Like MAE, but squares the errors.
* **Punishes big mistakes more**.

### Analogy:

Like crash tests — big crashes hurt more. MSE makes big misses count extra.

---

### E. **Root Mean Squared Error (RMSE)**

$$
\text{RMSE} = \sqrt{MSE}
$$

* It’s the **square root of MSE**, so you get the error in **original units** (like points or kg).
* Easier to interpret than MSE.

---

### F. **R-squared (R²)**

$$
R^2 = 1 - \frac{\text{SSE}}{\text{SST}}
$$

* Tells how much of the variation in the output is **explained by the model**.
* **R² = 1**: Perfect prediction
* **R² = 0**: No better than guessing the average

---

## 7. Evaluate the Model in Code

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(df['Scores'], df['Predicted'])
mae = mean_absolute_error(df['Scores'], df['Predicted'])
rmse = np.sqrt(mse)
r2 = r2_score(df['Scores'], df['Predicted'])

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
```

---

## 8. Summary Table of Metrics

| Metric       | Measures               | Interpretable? | Penalizes Big Errors? |
| ------------ | ---------------------- | -------------- | --------------------- |
| **Residual** | One prediction error   | Yes            | No                    |
| **SSE**      | Total squared error    | No             | Yes                   |
| **MAE**      | Average absolute error | Yes            | No                    |
| **MSE**      | Average squared error  | Not easily     | Yes                   |
| **RMSE**     | Root of MSE            | Yes            | Yes                   |
| **R²**       | % of output explained  | Yes            | Indirectly            |

---

Sure! Here's the section rewritten as a **Markdown table**:

---

## 9. Which Metrics Are Most Commonly Used and Why?

| **Metric**         | **Why It's Commonly Used**                                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| **MAE**            | Easy to understand. Directly tells you how much you're off on average. Often used in business settings where interpretability matters.  |
| **MSE**            | Preferred during model training since it penalizes big mistakes more heavily. Also mathematically easier to optimize (due to squaring). |
| **RMSE**           | Gives error in original units. Very useful for reporting performance to non-technical stakeholders.                                     |
| **R² (R-squared)** | A great summary score: "How much of the variation did my model explain?" Easy to compare between models.                                |
| **Residuals**      | Commonly plotted to check for patterns or assumptions in the model (e.g., non-linearity or outliers).                                   |

---

## 10. Final Thoughts

* **Simple Linear Regression** is your first step into machine learning.
* It’s perfect when there’s **one clear cause** and **one clear effect**.
* Understanding **residuals** and **evaluation metrics** is crucial for knowing how good your predictions are.

---

