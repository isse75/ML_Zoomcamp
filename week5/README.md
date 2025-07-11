# Chapter 5 - Deployment

[**Week 05 - Jupyter Notebook**](https://github.com/isse75/ML_Zoomcamp/blob/main/week5/Week-05.ipynb)
- Run Through of Saving and Loading Trained Model to .py File - [Week-05.py](https://github.com/isse75/ML_Zoomcamp/blob/main/week5/Week-05.py)
  
**[Train.py](https://github.com/isse75/ML_Zoomcamp/blob/main/week5/train.py) Python File**
- Running Training Model With some cues and confirmations
- Moved Parameters to top of File, making easier to make changes/adjustments
- Saving model

 **[Predict.py](https://github.com/isse75/ML_Zoomcamp/blob/main/week5/predict.py) Python File**
 - Running model on a new customer and seeing the predicted outcome

[**Ping.py Python File**](https://github.com/isse75/ML_Zoomcamp/blob/main/week5/ping.py)
Intro to Flask: Running a function on a web service (locally hosted) using Flask


📣 **KEY COMMAND TO RUN IN TERMINAL TO USE PRODUCTION SERVER** 
```
gunicorn --bind 0.0.0.0:9696 predict:app
```

**Pipfile Documents**
- Contains specific requirements required to run python project

:whale:**[Docker File](https://github.com/isse75/ML_Zoomcamp/blob/main/week5/Dockerfile)**
- Contains Settings Dependencies and commands required to run the project.
- Use the following commands in command line to run the Docker Container.
  
```
docker build -t zoomcamp-test .
docker run -it --rm -p 9696:9696 zoomcamp-test
```

