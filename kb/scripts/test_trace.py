import requests, json, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
resp = requests.post('http://localhost:8000/query', json={'collection_name': 'dev', 'query': 'What is the SABR model used for?'}, timeout=60)
d = resp.json()
print("Answer:", d.get('answer'))
print("\nTrace steps:")
for s in d.get('trace', []):
    print("  Step:", s.get('step_name'))
    print("    Description:", s.get('description'))
    print("    Input:", s.get('input'))
    print("    Output:", s.get('output'))

