import requests

host = 'heart-disease-env.eba-smhahyek.eu-west-1.elasticbeanstalk.com'
url = f'http://{host}/predict'


patient = {'age': 68,
 'sex': 1,
 'chest_pain_type': 2,
 'resting_bp': 118,
 'chol': 277,
 'fasting_blood_sugar': 0,
 'resting_ecg': 0,
 'max_hr_achieved': 151,
 'exercise_angina': 0,
 'st_depression': 1.0,
 'st_slope': 0,
 'no_vessels_fluoroscopy': 1,
 'thal_result': 3,
 'heart_disease': 0}


response = requests.post(url, json=patient).json()

if response['heart_disease'] == True:
    print('Sending promo email to %s' % ('xyz-123'))
else:
    print('not sending promo email to %s' % ('xyz-123'))
