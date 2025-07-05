# Midterm Project

**Command Line Code to Use Production Level Server When Running Predict App via Flask**

```
gunicorn --bind 0.0.0.0:9696 predict:app
```
**Command Line Code to Use Production Level Server When Running Predict App in PipEnv**

```
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```
![image](https://github.com/user-attachments/assets/60248697-5be5-48be-a1cc-4f8cba2ccf07)
