import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.stats import boxcox

import psycopg2 as pg2
from psycopg2.extras import RealDictCursor

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from statsmodels.formula.api import ols
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, SelectPercentile, f_classif
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from sqlalchemy import create_engine

import itertools

