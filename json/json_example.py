# Read json-file and print its data into string
import json

# Option 1: Read JSON-file
with open('YOUR_LOCAL_FILE_DIRECTORY/example.json') as f: 
  dataset = json.load(f)
print("Weather in", dataset["weather"][0]["city"],"is", dataset["weather"][0]["description"],"and the temperature is",dataset["weather"][0]["temperature"],"degrees of Celcius.")  

# Option 2: Create JSON without a file
dataset_json =  '''
{ "weather": [
        {
            "city": "Madrid",
            "temperature": "30",
            "description": "cloudy"
        },
        {
            "city": "Sevilla",
            "temperature": "32",
            "description": "sunny"
        }
    ]
                }
'''
dataset = json.loads(dataset_json)
print("Weather in", dataset["weather"][1]["city"],"is", dataset["weather"][1]["description"],"and the temperature is",dataset["weather"][1]["temperature"],"degrees of Celcius.")  
