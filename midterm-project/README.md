# Midterm Project

**Command Line Code to Use Production Level Server When Running Predict App via Flask**

```
gunicorn --bind 0.0.0.0:9696 predict:app
```
**Command Line Code to Use Production Level Server When Running Predict App in PipEnv**

```
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```
