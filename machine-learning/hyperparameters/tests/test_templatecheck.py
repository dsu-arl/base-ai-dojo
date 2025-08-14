from base_test import BaseTestValidator
import unittest


class TestTemplateCheck(BaseTestValidator):
    def test_success(self):
        script_content = """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print('Target Loss: 0.0252')
print(f' Model Loss: {loss:.4f}')
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._check_template()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
    
    def test_modified_template(self):
        cases = [
            {
                'desc': 'Missing load_iris',
                'content': """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print('Target Loss: 0.0252')
print(f' Model Loss: {loss:.4f}')
"""
            },
            {
                'desc': 'Missing train_test_split',
                'content': """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print('Target Loss: 0.0252')
print(f' Model Loss: {loss:.4f}')
"""
            },
            {
                'desc': 'Missing RandomForestClassifier',
                'content': """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print('Target Loss: 0.0252')
print(f' Model Loss: {loss:.4f}')
"""
            },
            {
                'desc': 'Missing model.fit',
                'content': """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print('Target Loss: 0.0252')
print(f' Model Loss: {loss:.4f}')
"""
            },
            {
                'desc': 'Missing model.predict_proba',
                'content': """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

loss = log_loss(y_test, y_pred_proba)
print('Target Loss: 0.0252')
print(f' Model Loss: {loss:.4f}')
"""
            },
            {
                'desc': 'Missing log_loss',
                'content': """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
print('Target Loss: 0.0252')
print(f' Model Loss: {loss:.4f}')
"""
            },
            {
                'desc': 'Missing print target loss',
                'content': """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print(f' Model Loss: {loss:.4f}')
"""
            },
            {
                'desc': 'Missing print model loss',
                'content': """
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the value of 'n_estimators' and find the value that
# results in a model loss of 0.0252
n_estimators = 123

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print('Target Loss: 0.0252')
"""
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                is_correct, msg = validator._check_template()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "No need to modify the given code template, only modify the 'n_estimators' value")


if __name__ == '__main__':
    unittest.main()