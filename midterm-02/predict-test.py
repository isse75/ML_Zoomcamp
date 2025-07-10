import requests

host = '34.248.88.252:9696'
url = f'http://{host}/predict'

customer = {
 'age': 32,
 'job': 'admin.',
 'marital': 'single',
 'education': 'tertiary',
 'default': 'no',
 'housing': 'yes',
 'loan': 'no',
 'contact': 'cellular',
 'month': 'jul',
 'day_of_week': 'thu',
 'campaign': 2,
 'previous': 1,
 'poutcome': 'success',
 'y': 0
}

response = requests.post(url, json=customer).json()

print(f"Customer deposit probability: {response['deposit_probability']:.3f}")

if response['will_deposit'] == True:
    print('Sending deposit offer to customer %s' % ('customer-123'))
else:
    print('Not sending deposit offer to customer %s' % ('customer-123'))