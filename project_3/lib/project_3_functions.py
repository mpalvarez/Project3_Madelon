{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%run __init__.py\n",
    "\n",
    "\n",
    "def fit_benchmark_model(X, y, model, name):\n",
    "    #train test split\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)\n",
    "    #scale the data\n",
    "    scaler = StandardScaler()\n",
    "    # Fit_transform\n",
    "    X_train_scaled = scaler.fit_transform(X_train)\n",
    "    # transform\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "    \n",
    "    model.fit(X_train_scaled, y_train)\n",
    "    return (name, model.score(X_train_scaled, y_train), model.score(X_test_scaled, y_test))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
